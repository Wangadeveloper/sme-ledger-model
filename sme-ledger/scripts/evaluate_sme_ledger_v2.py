#!/usr/bin/env python3
"""Comprehensive evaluation script for SME-Ledger V2 models (HuggingFace or GGUF).

Computes field-level accuracy, domain breakdowns, held-out benchmark accuracy,
capability response quality, and ledger running balance consistency.

Usage:
    python scripts/evaluate_sme_ledger_v2.py \
        --model-path models/merged \
        --data-dir data/sme_ledger_v2_dataset \
        --eval-limit 100 \
        --benchmark-limit 50 \
        --output-report reports/final_evaluation.json \
        --output-markdown reports/final_evaluation.md
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validator import DatasetValidator
from src.data.capability import CapabilityDatasetManager
from src.evaluation.parser import extract_json_value, canonical_value
from src.evaluation.field_metrics import FieldMetricsCalculator
from src.evaluation.capability_eval import CapabilityEvaluator
from src.evaluation.ledger_consistency import LedgerConsistencyEvaluator


def build_hf_generator(model_path: Path, max_new_tokens: int = 300) -> Tuple[Callable[[str], str], Any]:
    """Loads a Hugging Face model and creates a generation function."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading tokenizer from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it", use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Hugging Face model from {model_path}...")
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        with open(adapter_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base_name = cfg.get("base_model_name_or_path", "google/gemma-3-270m-it")
        base = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float32, low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(base, str(model_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch.float32, low_cpu_mem_usage=True)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    def generate(user_text: str) -> str:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_toks = output_tokens[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_toks, skip_special_tokens=True).strip()

    return generate, tokenizer


def build_gguf_generator(gguf_path: Path, llama_cli_path: Optional[Path] = None) -> Callable[[str], str]:
    """Creates a generation function using llama-cli on a quantized GGUF file."""
    cli_bin = llama_cli_path or Path("/home/shadeform/llama.cpp/build/bin/llama-cli")
    if not cli_bin.exists():
        raise FileNotFoundError(f"llama-cli binary not found at {cli_bin}")

    def generate(user_text: str) -> str:
        prompt = f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"
        cmd = [
            str(cli_bin),
            "-m", str(gguf_path),
            "-p", prompt,
            "-n", "300",
            "--temp", "0.0",
            "-t", "4",
            "--no-display-prompt",
            "--single-turn",
            "--simple-io",
            "--log-disable",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        raw = proc.stdout
        if "<start_of_turn>model" in raw:
            raw = raw.split("<start_of_turn>model")[-1]
        if "[ Prompt:" in raw:
            raw = raw.split("[ Prompt:")[0]
        if "Exiting..." in raw:
            raw = raw.split("Exiting...")[0]
        return raw.strip()

    return generate


def format_markdown_report(report: Dict[str, Any]) -> str:
    """Generates human-readable Markdown from evaluation report."""
    md = []
    md.append("# SME-Ledger V2 Evaluation Report\n")
    md.append(f"**Model Path**: `{report.get('model_path')}`  ")
    md.append(f"**Model Type**: `{report.get('model_type')}`  ")
    md.append(f"**Evaluated Examples**: {report.get('test_evaluation', {}).get('total_examples')} test records\n")

    test_ev = report.get("test_evaluation", {})
    md.append("## 1. Test Set Extraction Performance\n")
    md.append("| Metric | Score |")
    md.append("| :--- | :--- |")
    md.append(f"| **JSON Validity Rate** | `{test_ev.get('json_validity_rate', 0):.2%}` |")
    md.append(f"| **Exact Match Rate** | `{test_ev.get('exact_match_rate', 0):.2%}` |")
    md.append(f"| **Complete Record Accuracy** | `{test_ev.get('complete_record_accuracy', 0):.2%}` |\n")

    md.append("### Field-Level Accuracies\n")
    md.append("| Field | Accuracy |")
    md.append("| :--- | :--- |")
    for f, acc in sorted(test_ev.get("field_accuracies", {}).items()):
        name = f.replace("_accuracy", "")
        md.append(f"| `{name}` | `{acc:.2%}` |")
    md.append("")

    md.append("### Domain Breakdown\n")
    md.append("| Domain | Total | Correct | Accuracy |")
    md.append("| :--- | :--- | :--- | :--- |")
    for d, dinfo in sorted(test_ev.get("domain_breakdown", {}).items()):
        md.append(f"| `{d}` | {dinfo['total']} | {dinfo['correct']} | `{dinfo['accuracy']:.2%}` |")
    md.append("")

    # Benchmark
    bench_ev = report.get("benchmark_evaluation", {})
    md.append("## 2. Isolated Unseen Benchmark Performance\n")
    md.append("| Metric | Score |")
    md.append("| :--- | :--- |")
    md.append(f"| **Evaluated Records** | {bench_ev.get('total_examples', 0)} |")
    md.append(f"| **JSON Validity Rate** | `{bench_ev.get('json_validity_rate', 0):.2%}` |")
    md.append(f"| **Exact Match Rate** | `{bench_ev.get('exact_match_rate', 0):.2%}` |")
    md.append(f"| **Complete Record Accuracy** | `{bench_ev.get('complete_record_accuracy', 0):.2%}` |\n")

    # Capability
    cap_ev = report.get("capability_evaluation", {})
    md.append("## 3. Held-Out Capability & Privacy Evaluation\n")
    md.append(f"**Overall Capability Pass Rate**: `{cap_ev.get('capability_pass_rate', 0):.2%}` ({cap_ev.get('passed_count', 0)}/{cap_ev.get('total_questions', 0)} questions)\n")
    md.append("| Category | Questions | Passed | Pass Rate |")
    md.append("| :--- | :--- | :--- | :--- |")
    for c, cinfo in sorted(cap_ev.get("category_breakdown", {}).items()):
        md.append(f"| `{c}` | {cinfo['total']} | {cinfo['passed']} | `{cinfo['pass_rate']:.2%}` |")
    md.append("")

    # Ledger
    led_ev = report.get("ledger_consistency", {})
    md.append("## 4. Ledger Running Balance Consistency\n")
    md.append("| Metric | Value |")
    md.append("| :--- | :--- |")
    md.append(f"| **Reconcilable Transitions** | {led_ev.get('reconcilable_transitions', 0)} |")
    md.append(f"| **Balance Consistency Rate** | `{led_ev.get('balance_consistency_rate', 0):.2%}` |")
    md.append(f"| **Detected Inconsistencies** | {led_ev.get('inconsistent_transactions', 0)} |")
    fin = led_ev.get("financial_summary", {})
    md.append(f"| **Reconstructed Cash Flow** | Income: Ksh {fin.get('total_income', 0):,.2f} \| Expenses: Ksh {fin.get('total_expense', 0):,.2f} \| Net: Ksh {fin.get('net_cash_flow', 0):,.2f} |\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SME-Ledger V2 models on test, benchmark, capability, and ledger tasks.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to HuggingFace model or GGUF file")
    parser.add_argument("--data-dir", type=str, default="data/sme_ledger_v2_dataset", help="Path to dataset directory")
    parser.add_argument("--eval-limit", type=int, default=100, help="Maximum test examples to evaluate")
    parser.add_argument("--benchmark-limit", type=int, default=50, help="Maximum benchmark examples to evaluate")
    parser.add_argument("--output-report", type=str, default="reports/final_evaluation.json", help="Path to save JSON report")
    parser.add_argument("--output-markdown", type=str, default="reports/final_evaluation.md", help="Path to save Markdown report")
    args = parser.parse_args()

    model_p = Path(args.model_path).resolve()
    data_dir = Path(args.data_dir).resolve()

    validator = DatasetValidator(data_dir)
    test_records = validator.load_jsonl(data_dir / "test.jsonl")[: args.eval_limit]
    bench_records = validator.load_jsonl(data_dir / "benchmark.jsonl")[: args.benchmark_limit]

    # Setup generator
    if model_p.suffix.lower() == ".gguf":
        model_type = "GGUF"
        generate_fn = build_gguf_generator(model_p)
    else:
        model_type = "HuggingFace"
        generate_fn, _ = build_hf_generator(model_p)

    print("=" * 80)
    print(f"RUNNING EVALUATION ON {model_type}: {model_p.name}")
    print("=" * 80)

    # 1. Test set evaluation
    print(f"\n1. Evaluating on {len(test_records)} test examples...")
    test_preds = []
    extracted_records_for_ledger = []
    for r in test_records:
        user_msg = r["messages"][0]["content"]
        ref_text = r["messages"][1]["content"]
        raw_out = generate_fn(user_msg)
        parsed_pred = extract_json_value(raw_out)
        if parsed_pred is not None:
            extracted_records_for_ledger.append(parsed_pred)

        test_preds.append({
            "raw_output": raw_out,
            "reference_text": ref_text,
            "parsed_prediction": parsed_pred,
            "parsed_reference": extract_json_value(ref_text),
        })

    metrics_calc = FieldMetricsCalculator()
    test_metrics = metrics_calc.evaluate_predictions(test_preds)

    # 2. Benchmark evaluation
    print(f"\n2. Evaluating on {len(bench_records)} unseen benchmark examples...")
    bench_preds = []
    for r in bench_records:
        user_msg = r["messages"][0]["content"]
        ref_text = r["messages"][1]["content"]
        raw_out = generate_fn(user_msg)
        bench_preds.append({
            "raw_output": raw_out,
            "reference_text": ref_text,
            "parsed_prediction": extract_json_value(raw_out),
            "parsed_reference": extract_json_value(ref_text),
        })
    bench_metrics = metrics_calc.evaluate_predictions(bench_preds)

    # 3. Held-out capability evaluation
    print("\n3. Evaluating on held-out capability questions...")
    cap_mgr = CapabilityDatasetManager(data_dir / "capability_examples.csv")
    _, held_out_cap = cap_mgr.create_stratified_split(held_out_per_category=1)
    cap_evaluator = CapabilityEvaluator(held_out_cap)
    cap_metrics = cap_evaluator.evaluate(generate_fn)

    # 4. Ledger consistency
    print("\n4. Evaluating ledger running balance consistency...")
    ledger_evaluator = LedgerConsistencyEvaluator()
    ledger_df = ledger_evaluator.build_ledger_dataframe(extracted_records_for_ledger)
    ledger_metrics = ledger_evaluator.evaluate_ledger_consistency(ledger_df)

    report = {
        "model_path": str(model_p),
        "model_type": model_type,
        "test_evaluation": test_metrics,
        "benchmark_evaluation": bench_metrics,
        "capability_evaluation": cap_metrics,
        "ledger_consistency": ledger_metrics,
    }

    # Save JSON report
    out_rep = Path(args.output_report).resolve()
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rep, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Evaluation report saved to {out_rep}")

    # Save Markdown report
    md_content = format_markdown_report(report)
    out_md = Path(args.output_markdown).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"✓ Markdown report saved to {out_md}")

    print("\n" + "=" * 80)
    print(f"JSON Validity       : {test_metrics['json_validity_rate']:.2%}")
    print(f"Complete Record Acc : {test_metrics['complete_record_accuracy']:.2%}")
    print(f"Benchmark Acc       : {bench_metrics['complete_record_accuracy']:.2%}")
    print(f"Capability Pass Rate: {cap_metrics['capability_pass_rate']:.2%}")
    print(f"Ledger Consistency  : {ledger_metrics['balance_consistency_rate']:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
