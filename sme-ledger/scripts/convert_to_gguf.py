#!/usr/bin/env python3
"""GGUF conversion and Q4_K_M quantization script for SME-Ledger V2.

Usage:
    python scripts/convert_to_gguf.py \
        --merged-model models/merged \
        --output-dir models/gguf \
        --llama-cpp-dir /home/shadeform/llama.cpp \
        --quant-type Q4_K_M \
        --run-name sme-ledger-v2
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.config import TrainingPipelineConfig
from src.export.converter import GGUFConverter
from src.export.quantizer import GGUFQuantizer
from src.export.merger import ModelMerger


def main():
    parser = argparse.ArgumentParser(description="Convert and quantize SME-Ledger V2 model to GGUF format.")
    parser.add_argument("--merged-model", type=str, default=None, help="Path to pre-merged model directory")
    parser.add_argument("--lora-adapter", type=str, default=None, help="Path to LoRA adapter to merge first if not merged")
    parser.add_argument("--base-model", type=str, default="google/gemma-3-270m-it", help="Base model identifier")
    parser.add_argument("--output-dir", type=str, default="models/gguf", help="Output directory for GGUF files")
    parser.add_argument("--llama-cpp-dir", type=str, default="/home/shadeform/llama.cpp", help="Path to llama.cpp repository")
    parser.add_argument("--quant-type", type=str, default="Q4_K_M", help="Quantization type (e.g. Q4_K_M)")
    parser.add_argument("--run-name", type=str, default="sme-ledger-v2", help="Model run name prefix")
    args = parser.parse_args()

    config = TrainingPipelineConfig(
        base_model=args.base_model,
        llama_cpp_dir=args.llama_cpp_dir,
        run_name=args.run_name,
        quantization_type=args.quant_type,
    )

    merged_path = Path(args.merged_model) if args.merged_model else None

    # Merge LoRA if needed
    if not merged_path or not merged_path.exists():
        if args.lora_adapter and Path(args.lora_adapter).exists():
            print(f"Merging LoRA adapter from {args.lora_adapter}...")
            merger = ModelMerger(config)
            merged_path = merger.merge_and_export(
                base_model_name_or_path=args.base_model,
                lora_adapter_path=args.lora_adapter,
                output_dir="models/merged",
            )
        elif Path(config.merged_dir).exists():
            merged_path = Path(config.merged_dir)
        else:
            raise FileNotFoundError(
                f"No merged model found at '{args.merged_model}' and no valid --lora-adapter specified."
            )

    converter = GGUFConverter(config)
    quantizer = GGUFQuantizer(config)

    # 1. Convert to FP16 GGUF
    f16_gguf = converter.convert_to_f16_gguf(
        merged_dir=merged_path,
        output_gguf_path=Path(args.output_dir) / f"{args.run_name}-f16.gguf",
    )

    # 2. Quantize to Q4_K_M
    q4_gguf = quantizer.quantize(
        f16_gguf_path=f16_gguf,
        output_q4_path=Path(args.output_dir) / f"{args.run_name}-{args.quant_type}.gguf",
        quant_type=args.quant_type,
    )

    # 3. Validate runtime inference
    print("\nValidating quantized model inference...")
    val_res = quantizer.validate_gguf_inference(q4_gguf)

    print("\n" + "=" * 80)
    print("CONVERSION & QUANTIZATION COMPLETE")
    print(f"FP16 GGUF : {f16_gguf}")
    print(f"Q4 GGUF   : {q4_gguf}")
    print(f"Size      : {round(q4_gguf.stat().st_size / (1024**2), 2)} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
