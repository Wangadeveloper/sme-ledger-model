#!/usr/bin/env python3
"""Master end-to-end training and export pipeline for SME-Ledger V2.

Executes the entire lifecycle:
DATASET
  → validation
  → dataset balancing/inspection
  → instruction-tuning dataset preparation (with capability mixture)
  → base model loading
  → LoRA/PEFT fine-tuning
  → validation/evaluation
  → checkpoint saving
  → LoRA adapter export
  → merged model export
  → GGUF conversion
  → Q4_K_M quantization
  → final model validation
  → benchmark/capability evaluation
  → training manifest and model card report

Usage:
    python scripts/train_sme_ledger_v2.py --stage all
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.config import TrainingPipelineConfig
from src.data.validator import DatasetValidator
from src.data.dataset import SmeDatasetBuilder
from src.data.capability import CapabilityDatasetManager
from src.training.trainer import SmeTrainer
from src.training.hardware import HardwareInfo, cleanup_memory, print_memory
from src.export.merger import ModelMerger
from src.export.converter import GGUFConverter
from src.export.quantizer import GGUFQuantizer
from src.evaluation.field_metrics import FieldMetricsCalculator
from src.evaluation.capability_eval import CapabilityEvaluator
from src.evaluation.ledger_consistency import LedgerConsistencyEvaluator
from src.evaluation.parser import extract_json_value


def generate_model_card(
    config: TrainingPipelineConfig,
    dataset_report: Dict[str, Any],
    eval_report: Dict[str, Any],
    gguf_path: Optional[Path] = None,
    output_path: Optional[Path | str] = None,
) -> str:
    """Generates the comprehensive model card markdown document."""
    card = []
    card.append("# SME-Ledger V2 Model Card\n")
    card.append("## Overview\n")
    card.append(
        "SME-Ledger V2 is an offline-first, on-device financial intelligence model designed for "
        "Small and Medium Enterprises (SMEs) in East Africa. It extracts structured accounting records from "
        "raw mobile money (M-PESA) and banking SMS notifications, reconciles running cash balances, and answers "
        "capability and privacy questions without leaking data to cloud endpoints.\n"
    )

    card.append("## Model Details\n")
    card.append(f"- **Base Model**: `{config.base_model}`")
    card.append(f"- **Fine-Tuning Method**: LoRA (PEFT)")
    card.append(f"- **LoRA Parameters**: Rank={config.lora_r}, Alpha={config.lora_alpha}, Dropout={config.lora_dropout}")
    card.append(f"- **Quantization**: `{config.quantization_type}` (GGUF format)")
    if gguf_path and gguf_path.exists():
        size_mb = round(gguf_path.stat().st_size / (1024**2), 2)
        card.append(f"- **Final GGUF File**: `{gguf_path.name}` ({size_mb} MB)")
    card.append(f"- **Deployment Target**: Edge mobile / low-resource on-device runtime (`llama.cpp`)\n")

    card.append("## Dataset\n")
    splits = dataset_report.get("splits", {})
    card.append(f"- **Train Count**: {splits.get('train', {}).get('count', 'N/A'):,} examples")
    card.append(f"- **Validation Count**: {splits.get('validation', {}).get('count', 'N/A'):,} examples")
    card.append(f"- **Test Count**: {splits.get('test', {}).get('count', 'N/A'):,} examples")
    card.append(f"- **Benchmark Count**: {splits.get('benchmark', {}).get('count', 'N/A'):,} examples (strictly isolated)")
    card.append(f"- **Capability Instruction Queries**: {dataset_report.get('capability_examples', {}).get('count', 0)} examples across 10 categories\n")

    card.append("## Training Hyperparameters\n")
    card.append(f"- **Epochs**: {config.num_epochs}")
    card.append(f"- **Learning Rate**: {config.learning_rate}")
    card.append(f"- **Batch Size**: {config.train_batch_size} (effective: {config.train_batch_size * config.gradient_accumulation_steps})")
    card.append(f"- **Sequence Length**: {config.max_length}")
    card.append(f"- **Random Seed**: {config.seed}\n")

    card.append("## Evaluation Performance\n")
    test_ev = eval_report.get("test_evaluation", {})
    bench_ev = eval_report.get("benchmark_evaluation", {})
    cap_ev = eval_report.get("capability_evaluation", {})
    led_ev = eval_report.get("ledger_consistency", {})

    card.append("| Task / Metric | Result |")
    card.append("| :--- | :--- |")
    card.append(f"| **JSON Validity Rate** | `{test_ev.get('json_validity_rate', 0):.2%}` |")
    card.append(f"| **Complete Record Accuracy** | `{test_ev.get('complete_record_accuracy', 0):.2%}` |")
    card.append(f"| **Isolated Benchmark Accuracy** | `{bench_ev.get('complete_record_accuracy', 0):.2%}` |")
    card.append(f"| **Held-Out Capability Pass Rate** | `{cap_ev.get('capability_pass_rate', 0):.2%}` |")
    card.append(f"| **Ledger Balance Consistency** | `{led_ev.get('balance_consistency_rate', 0):.2%}` |\n")

    card.append("## Intended Use & Safety\n")
    card.append(
        "- **Primary Use**: Local, offline parsing of M-PESA and banking SMS receipts into double-entry accounting schemas.\n"
        "- **Privacy**: Zero external network requests during inference; sensitive financial data remains strictly on-device.\n"
        "- **Limitations**: Does not execute financial transactions; operates in read-only analytical mode."
    )

    card_text = "\n".join(card)
    if output_path:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(card_text)
        print(f"✓ Model card saved to {out_p}")

    return card_text


def run_pipeline(args):
    """Executes requested stages of the pipeline."""
    config = TrainingPipelineConfig(
        base_model=args.base_model,
        hf_token=args.hf_token,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        run_name=args.run_name,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        llama_cpp_dir=args.llama_cpp_dir,
        seed=args.seed,
    )

    # Save run configuration
    config.manifests_dir.mkdir(parents=True, exist_ok=True)
    config.save_json(config.manifests_dir / "run_config.json")

    stage = args.stage.lower()
    run_all = stage == "all"

    print("=" * 80)
    print("SME-LEDGER V2 MASTER PIPELINE")
    print(f"Stage       : {stage}")
    print(f"Base Model  : {config.base_model}")
    print(f"Data Dir    : {config.data_dir}")
    print(f"Output Dir  : {config.output_dir}")
    print(f"Seed        : {config.seed}")
    print("=" * 80)

    dataset_report_path = Path(config.reports_dir) / "dataset_report.json"
    dataset_report: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # STAGE 1: VALIDATE
    # -------------------------------------------------------------------------
    if run_all or stage == "validate":
        print("\n>>> STAGE: VALIDATING DATASET")
        validator = DatasetValidator(config.data_dir)
        dataset_report = validator.run_full_validation(report_path=dataset_report_path)
    elif dataset_report_path.exists():
        with open(dataset_report_path, "r", encoding="utf-8") as f:
            dataset_report = json.load(f)

    # -------------------------------------------------------------------------
    # STAGE 2: PREPARE DATASET
    # -------------------------------------------------------------------------
    train_ds, val_ds = None, None
    if run_all or stage in ["prepare", "train"]:
        print("\n>>> STAGE: PREPARING DATASET")
        trainer = SmeTrainer(config)
        tok = trainer.load_tokenizer()

        builder = SmeDatasetBuilder(
            data_dir=config.data_dir,
            tokenizer=tok,
            max_length=config.max_length,
            seed=config.seed,
            capability_ratio=config.capability_ratio,
        )

        held_out_cap_path = Path(config.reports_dir) / "held_out_capability_test.jsonl"
        train_ds, val_ds, test_ds, bench_ds, held_out_cap = builder.load_and_prepare(
            include_capability=True, save_held_out_path=held_out_cap_path
        )

        if stage == "prepare":
            print("✓ Dataset preparation stage complete.")
            return

    # -------------------------------------------------------------------------
    # STAGE 3: TRAIN LoRA
    # -------------------------------------------------------------------------
    if run_all or stage == "train":
        print("\n>>> STAGE: LoRA FINE-TUNING")
        train_metrics, eval_metrics = trainer.train(train_ds, val_ds)

    # -------------------------------------------------------------------------
    # STAGE 4: MERGE LoRA
    # -------------------------------------------------------------------------
    merged_path = Path(config.merged_dir)
    if run_all or stage == "merge":
        print("\n>>> STAGE: MERGING LoRA INTO BASE MODEL")
        merger = ModelMerger(config)
        merged_path = merger.merge_and_export(
            base_model_name_or_path=config.base_model,
            lora_adapter_path=config.lora_dir,
            output_dir=config.merged_dir,
        )

    # -------------------------------------------------------------------------
    # STAGE 5: QUANTIZE TO GGUF
    # -------------------------------------------------------------------------
    f16_gguf_path = Path(config.gguf_dir) / f"{config.run_name}-f16.gguf"
    q4_gguf_path = Path(config.gguf_dir) / f"{config.run_name}-{config.quantization_type}.gguf"

    if run_all or stage == "quantize":
        print("\n>>> STAGE: GGUF CONVERSION & Q4_K_M QUANTIZATION")
        converter = GGUFConverter(config)
        quantizer = GGUFQuantizer(config)

        f16_gguf_path = converter.convert_to_f16_gguf(
            merged_dir=merged_path,
            output_gguf_path=f16_gguf_path,
        )

        q4_gguf_path = quantizer.quantize(
            f16_gguf_path=f16_gguf_path,
            output_q4_path=q4_gguf_path,
            quant_type=config.quantization_type,
        )

        print("\nValidating quantized GGUF inference...")
        quantizer.validate_gguf_inference(q4_gguf_path)

    # -------------------------------------------------------------------------
    # STAGE 6: EVALUATE & BENCHMARK
    # -------------------------------------------------------------------------
    if run_all or stage in ["evaluate", "benchmark"]:
        print("\n>>> STAGE: EVALUATION & BENCHMARK")
        eval_report_path = Path(config.reports_dir) / "final_evaluation.json"
        eval_markdown_path = Path(config.reports_dir) / "final_evaluation.md"
        model_card_path = Path(config.reports_dir) / "model_card.md"

        target_eval_model = q4_gguf_path if q4_gguf_path.exists() else merged_path

        eval_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_sme_ledger_v2.py"),
            "--model-path", str(target_eval_model),
            "--data-dir", str(config.data_dir),
            "--eval-limit", str(config.eval_limit),
            "--benchmark-limit", str(config.benchmark_eval_limit),
            "--output-report", str(eval_report_path),
            "--output-markdown", str(eval_markdown_path),
        ]

        subprocess.run(eval_cmd, check=True)

        if eval_report_path.exists():
            with open(eval_report_path, "r", encoding="utf-8") as f:
                eval_report = json.load(f)

            generate_model_card(
                config=config,
                dataset_report=dataset_report,
                eval_report=eval_report,
                gguf_path=q4_gguf_path if q4_gguf_path.exists() else None,
                output_path=model_card_path,
            )

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION FINISHED SUCCESSFULLY")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Master training pipeline for SME-Ledger V2.")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["validate", "prepare", "train", "merge", "quantize", "evaluate", "benchmark", "all"],
        help="Pipeline stage to execute",
    )
    parser.add_argument("--base-model", type=str, default="google/gemma-3-270m-it", help="Base model identifier")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face authentication token")
    parser.add_argument("--data-dir", type=str, default="data/sme_ledger_v2_dataset", help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for models and checkpoints")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Reports directory")
    parser.add_argument("--run-name", type=str, default="sme-ledger-v2", help="Run identifier")
    parser.add_argument("--llama-cpp-dir", type=str, default="/home/shadeform/llama.cpp", help="Path to llama.cpp repo")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=20260910, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
