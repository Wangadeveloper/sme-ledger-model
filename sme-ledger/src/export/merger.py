"""Merges LoRA adapter into base model and exports full HuggingFace model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.training.config import TrainingPipelineConfig
from src.training.hardware import HardwareInfo, check_model_parameters, cleanup_memory


class ModelMerger:
    """Merges trained LoRA adapter weights into the base model."""

    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        self.hardware = HardwareInfo.detect()

    def merge_and_export(
        self,
        base_model_name_or_path: Optional[str] = None,
        lora_adapter_path: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
    ) -> Path:
        """Merges LoRA weights and saves the combined model to disk."""
        base_name = base_model_name_or_path or self.config.base_model
        adapter_path = Path(lora_adapter_path or self.config.lora_dir).resolve()
        out_dir = Path(output_dir or self.config.merged_dir).resolve()

        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter path not found: {adapter_path}")

        cleanup_memory()
        print(f"Loading base model: {base_name}")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_name,
                dtype=self.hardware.selected_dtype,
                low_cpu_mem_usage=True,
                token=self.config.hf_token,
            )
        except Exception as e:
            print(f"Could not load {base_name} directly ({e}), trying fallback: {self.config.fallback_base_model}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.fallback_base_model,
                dtype=self.hardware.selected_dtype,
                low_cpu_mem_usage=True,
                token=self.config.hf_token,
            )

        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))

        print("Merging LoRA adapter into base weights...")
        model.eval()
        merged_model = model.merge_and_unload()
        merged_model.config.use_cache = True

        check_model_parameters(merged_model, "MERGED MODEL")

        # Load and save tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_name, token=self.config.hf_token, use_fast=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Test inference verification
        print("Running pre-export sanity inference on merged model...")
        test_msg = [{"role": "user", "content": "What is SME-Ledger?"}]
        prompt = tokenizer.apply_chat_template(test_msg, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        if self.hardware.cuda_available:
            merged_model = merged_model.cuda()
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            out_tokens = merged_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(out_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"✓ Sanity response: {gen_text.strip()[:100]}...")

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving merged model to: {out_dir}")
        merged_model.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)

        print(f"✓ Merged model successfully saved to {out_dir}")
        return out_dir
