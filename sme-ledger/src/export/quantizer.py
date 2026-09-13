"""Quantizes GGUF model to Q4_K_M and runs final on-device validation using llama.cpp."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from src.training.config import TrainingPipelineConfig


class GGUFQuantizer:
    """Manages Q4_K_M quantization and validates inference execution via llama.cpp."""

    def __init__(self, config: TrainingPipelineConfig):
        self.config = config

    def find_llama_quantize(self) -> Path:
        """Locates llama-quantize binary."""
        candidates = [
            Path(self.config.llama_cpp_dir) / "build" / "bin" / "llama-quantize",
            Path("/home/shadeform/llama.cpp/build/bin/llama-quantize"),
            Path("./llama.cpp/build/bin/llama-quantize"),
            Path(self.config.llama_cpp_dir) / "llama-quantize",
        ]

        for c in candidates:
            if c.exists():
                return c.resolve()

        which_q = shutil.which("llama-quantize")
        if which_q:
            return Path(which_q).resolve()

        raise FileNotFoundError(
            f"llama-quantize executable was not found. Looked in: {[str(c) for c in candidates]}"
        )

    def find_llama_cli(self) -> Optional[Path]:
        """Locates llama-cli binary for test execution."""
        candidates = [
            Path(self.config.llama_cpp_dir) / "build" / "bin" / "llama-cli",
            Path("/home/shadeform/llama.cpp/build/bin/llama-cli"),
            Path("./llama.cpp/build/bin/llama-cli"),
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        which_cli = shutil.which("llama-cli")
        return Path(which_cli).resolve() if which_cli else None

    def quantize(
        self,
        f16_gguf_path: Path | str,
        output_q4_path: Optional[Path | str] = None,
        quant_type: str = "Q4_K_M",
    ) -> Path:
        """Quantizes an FP16 GGUF file to Q4_K_M."""
        f16_p = Path(f16_gguf_path).resolve()
        if not f16_p.exists():
            raise FileNotFoundError(f"Input FP16 GGUF does not exist: {f16_p}")

        quantizer_bin = self.find_llama_quantize()
        out_q4 = Path(
            output_q4_path
            or (self.config.gguf_dir / f"{self.config.run_name}-{quant_type}.gguf")
        ).resolve()

        cmd = [
            str(quantizer_bin),
            str(f16_p),
            str(out_q4),
            quant_type,
        ]

        print("=" * 80)
        print(f"QUANTIZING GGUF TO {quant_type}")
        print("Command:", " ".join(cmd))
        print("=" * 80)

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        print(result.stdout)

        if result.returncode != 0:
            raise RuntimeError(
                f"llama-quantize failed with code {result.returncode}.\nOutput:\n{result.stdout}"
            )

        if not out_q4.exists():
            raise FileNotFoundError(f"Quantized GGUF was not found at {out_q4}")

        size_mb = round(out_q4.stat().st_size / (1024**2), 2)
        print(f"✓ Final quantized GGUF created: {out_q4} ({size_mb} MB)")
        return out_q4

    def validate_gguf_inference(
        self,
        gguf_model_path: Path | str,
        test_prompts: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """Runs test inference on the quantized GGUF model using llama-cli."""
        model_p = Path(gguf_model_path).resolve()
        llama_cli = self.find_llama_cli()

        if not llama_cli or not llama_cli.exists():
            print("⚠ llama-cli not found, skipping runtime GGUF generation test.")
            return {"tested": False, "reason": "llama-cli not found"}

        prompts = test_prompts or [
            "UIPB1SKBO1 Confirmed. You have received Ksh18,810.00 from Victor Kiptoo 0796617790 on 23/4/26 at 03:41. New M-PESA balance is Ksh108,890.68.",
            "What can you do?",
            "Can this work without sending my data to the cloud?",
        ]

        results = []
        for prompt_text in prompts:
            formatted_prompt = f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
            cmd = [
                str(llama_cli),
                "-m", str(model_p),
                "-p", formatted_prompt,
                "-n", "150",
                "--temp", "0.0",
                "-t", "4",
                "--no-display-prompt",
                "--single-turn",
                "--simple-io",
                "--log-disable",
            ]

            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = proc.stdout
            if "<start_of_turn>model" in output:
                output = output.split("<start_of_turn>model")[-1]
            if "[ Prompt:" in output:
                output = output.split("[ Prompt:")[0]
            if "Exiting..." in output:
                output = output.split("Exiting...")[0]
            output = output.strip()
            results.append({
                "prompt": prompt_text,
                "output": output,
                "exit_code": proc.returncode,
            })
            print(f"\n[GGUF TEST] Prompt: {prompt_text}")
            print(f"[GGUF TEST] Response: {output[:200]}...")

        return {
            "tested": True,
            "model_path": str(model_p),
            "size_mb": round(model_p.stat().st_size / (1024**2), 2),
            "results": results,
        }
