"""Trainer class implementing PEFT/LoRA fine-tuning and validation for SME-Ledger V2."""

from __future__ import annotations

import datetime
import inspect
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.training.config import TrainingPipelineConfig
from src.training.collator import CausalLMDataCollator
from src.training.hardware import (
    HardwareInfo,
    check_model_parameters,
    cleanup_memory,
    print_memory,
)


class SmeTrainer:
    """Manages model loading, LoRA configuration, training arguments, training loop, and adapter saving."""

    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        self.hardware = HardwareInfo.detect()
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.base_model: Optional[PreTrainedModel] = None
        self.peft_model: Optional[PeftModel] = None
        self.trainer: Optional[Trainer] = None
        self.collator: Optional[CausalLMDataCollator] = None

        # Set seeds
        set_seed(self.config.seed)

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        """Loads and configures the tokenizer."""
        print(f"Loading tokenizer: {self.config.base_model}")
        try:
            tok = AutoTokenizer.from_pretrained(
                self.config.base_model,
                token=self.config.hf_token,
                use_fast=True,
            )
        except Exception as e:
            print(f"Notice: Could not load tokenizer from {self.config.base_model} ({e})")
            print(f"Trying fallback tokenizer: {self.config.fallback_base_model}")
            tok = AutoTokenizer.from_pretrained(
                self.config.fallback_base_model,
                token=self.config.hf_token,
                use_fast=True,
            )

        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        self.tokenizer = tok
        self.collator = CausalLMDataCollator(tok)
        return tok

    def load_base_model(self) -> PreTrainedModel:
        """Loads base causal language model, validating parameters and performing forward test."""
        if self.tokenizer is None:
            self.load_tokenizer()

        cleanup_memory()
        print(f"Loading base model (target dtype: {self.hardware.selected_dtype})...")

        model_name = self.config.base_model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=self.hardware.selected_dtype,
                low_cpu_mem_usage=True,
                token=self.config.hf_token,
            )
        except Exception as e:
            print(f"Notice: Could not load base model '{model_name}': {e}")
            print(f"Falling back to public mirror: '{self.config.fallback_base_model}'")
            model_name = self.config.fallback_base_model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=self.hardware.selected_dtype,
                low_cpu_mem_usage=True,
                token=self.config.hf_token,
            )

        if self.hardware.cuda_available:
            model = model.cuda()

        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.config.use_cache = False

        check_model_parameters(model, f"BASE MODEL ({model_name})")
        print_memory("BASE MODEL LOADED")
        self.base_model = model
        return model

    def setup_lora(self) -> PeftModel:
        """Applies LoRA adapter to the base model and checks gradients."""
        if self.base_model is None:
            self.load_base_model()

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config.lora_target_modules,
        )

        print("Initializing LoRA adapter...")
        peft_m = get_peft_model(self.base_model, lora_config)
        peft_m.print_trainable_parameters()
        check_model_parameters(peft_m, "LoRA MODEL")

        self.peft_model = peft_m
        return peft_m

    def test_forward_backward(self, sample_batch: Dict[str, torch.Tensor]) -> None:
        """Validates gradient computation on a test batch."""
        if self.peft_model is None:
            raise RuntimeError("LoRA model not initialized.")

        device = next(self.peft_model.parameters()).device
        batch = {k: v.to(device) for k, v in sample_batch.items()}

        self.peft_model.train()
        self.peft_model.zero_grad(set_to_none=True)

        outputs = self.peft_model(**batch)
        loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(f"Sanity check forward loss is non-finite: {loss}")

        loss.backward()

        total_grad_norm_sq = 0.0
        for name, param in self.peft_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    raise RuntimeError(f"Non-finite gradient in {name}")
                total_grad_norm_sq += param.grad.detach().float().norm(2).item() ** 2

        self.peft_model.zero_grad(set_to_none=True)
        print(f"✓ Gradient sanity check passed (grad norm: {math.sqrt(total_grad_norm_sq):.4f})")

    def build_training_arguments(self, num_train_examples: int) -> TrainingArguments:
        """Constructs version-adaptive TrainingArguments for Transformers 4.x / 5.x."""
        checkpoint_dir = self.config.checkpoints_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        ta_params = inspect.signature(TrainingArguments.__init__).parameters

        kwargs: Dict[str, Any] = {
            "output_dir": str(checkpoint_dir),
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": self.config.train_batch_size,
            "per_device_eval_batch_size": self.config.eval_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "max_grad_norm": self.config.max_grad_norm,
            "logging_steps": 10,
            "save_total_limit": 2,
            "report_to": "none",
            "remove_unused_columns": False,
            "seed": self.config.seed,
            "data_seed": self.config.seed,
        }

        # Precision flags
        if self.hardware.cuda_available:
            if self.hardware.selected_dtype == torch.bfloat16:
                kwargs["bf16"] = True
                kwargs["fp16"] = False
            else:
                kwargs["bf16"] = False
                kwargs["fp16"] = False
        else:
            kwargs["bf16"] = False
            kwargs["fp16"] = False

        # Warmup
        if "warmup_ratio" in ta_params:
            kwargs["warmup_ratio"] = self.config.warmup_ratio
        elif "warmup_steps" in ta_params:
            steps_per_epoch = math.ceil(
                num_train_examples
                / (self.config.train_batch_size * self.config.gradient_accumulation_steps)
            )
            total_steps = steps_per_epoch * self.config.num_epochs
            kwargs["warmup_steps"] = max(1, int(total_steps * self.config.warmup_ratio))

        # Evaluation & Save Strategy
        strat_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
        if strat_key in ta_params:
            kwargs[strat_key] = "epoch"
        if "save_strategy" in ta_params:
            kwargs["save_strategy"] = "epoch"
        if "load_best_model_at_end" in ta_params:
            kwargs["load_best_model_at_end"] = True
        if "metric_for_best_model" in ta_params:
            kwargs["metric_for_best_model"] = "eval_loss"
        if "greater_is_better" in ta_params:
            kwargs["greater_is_better"] = False

        return TrainingArguments(**kwargs)

    def train(
        self, train_dataset: Dataset, val_dataset: Dataset
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Runs the complete LoRA training loop and validation."""
        if self.peft_model is None:
            self.setup_lora()

        # Sanity test batch
        test_batch = self.collator([train_dataset[0]])
        self.test_forward_backward(test_batch)

        training_args = self.build_training_arguments(len(train_dataset))

        self.trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.collator,
        )

        print("=" * 80)
        print("STARTING SME-LEDGER V2 LoRA TRAINING")
        print("=" * 80)
        print_memory("BEFORE TRAINING")

        start_time = time.time()
        latest_checkpoint = get_last_checkpoint(str(self.config.checkpoints_dir))

        if latest_checkpoint:
            print(f"\n[RESUME] Latest checkpoint found: {latest_checkpoint}")
            print("[RESUME] Restoring model, optimizer, scheduler, RNG state, and trainer state...")
            train_result = self.trainer.train(
                resume_from_checkpoint=latest_checkpoint
            )
        else:
            print("\n[TRAIN] No checkpoint found. Starting training from scratch.")
            train_result = self.trainer.train()
        train_duration = round(time.time() - start_time, 2)

        print_memory("AFTER TRAINING")
        print(f"✓ Training finished in {train_duration}s. Final train loss: {train_result.training_loss:.4f}")

        # Validation
        print("Running validation...")
        eval_metrics = self.trainer.evaluate()
        print(f"Validation metrics: {json.dumps(eval_metrics, indent=2)}")

        # Save LoRA adapter and tokenizer
        lora_out = self.config.lora_dir
        lora_out.mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(lora_out)
        self.tokenizer.save_pretrained(lora_out)
        print(f"✓ LoRA adapter saved to {lora_out}")

        # Git commit if available
        git_commit = "unknown"
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            pass

        # Manifest
        manifest = {
            "run_name": self.config.run_name,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "git_commit": git_commit,
            "base_model": self.config.base_model,
            "hardware": self.hardware.summary(),
            "config": self.config.to_dict(),
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset),
            "train_duration_seconds": train_duration,
            "train_loss": train_result.training_loss,
            "eval_metrics": eval_metrics,
            "artifacts": {
                "lora_dir": str(lora_out),
            },
        }

        self.config.manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.config.manifests_dir / "training_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"✓ Training manifest saved to {manifest_path}")

        return train_result.metrics, eval_metrics
