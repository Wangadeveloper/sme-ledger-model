"""Converts merged Hugging Face model to FP16 GGUF format using llama.cpp."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download
from src.training.config import TrainingPipelineConfig


class GGUFConverter:
    """Handles preparation and conversion of Hugging Face weights to GGUF format."""

    def __init__(self, config: TrainingPipelineConfig):
        self.config = config

    def find_llama_cpp_converter(self) -> Path:
        """Locates convert_hf_to_gguf.py in configured directory or system paths."""
        candidates = [
            Path(self.config.llama_cpp_dir) / "convert_hf_to_gguf.py",
            Path("/home/shadeform/llama.cpp/convert_hf_to_gguf.py"),
            Path("./llama.cpp/convert_hf_to_gguf.py"),
            Path("/kaggle/working/llama.cpp/convert_hf_to_gguf.py"),
        ]

        for c in candidates:
            if c.exists():
                return c.resolve()

        # Check system PATH
        which_conv = shutil.which("convert_hf_to_gguf.py")
        if which_conv:
            return Path(which_conv).resolve()

        raise FileNotFoundError(
            f"convert_hf_to_gguf.py was not found. Looked in: {[str(c) for c in candidates]}. "
            "Specify valid --llama-cpp-dir."
        )

    def ensure_sentencepiece_tokenizer(
        self, target_dir: Path, base_model_id: str
    ) -> Path:
        """Ensures tokenizer.model is available for Gemma-3 conversion."""
        dest_spm = target_dir / "tokenizer.model"
        if dest_spm.exists():
            return dest_spm

        # Search merged dir
        merged_spm = Path(self.config.merged_dir) / "tokenizer.model"
        if merged_spm.exists():
            shutil.copy2(merged_spm, dest_spm)
            return dest_spm

        # Search local filesystem / Kaggle inputs
        search_roots = [
            Path("~/.cache/huggingface/hub").expanduser(),
            Path("/kaggle/input"),
        ]
        for root in search_roots:
            if not root.exists():
                continue
            for match in root.rglob("tokenizer.model"):
                if "gemma" in str(match).lower() or match.stat().st_size > 1_000_000:
                    shutil.copy2(match, dest_spm)
                    print(f"✓ Copied tokenizer.model from local cache: {match}")
                    return dest_spm

        # Download using huggingface_hub
        print(f"Downloading tokenizer.model for {base_model_id}...")
        for repo_id in [base_model_id, self.config.fallback_base_model]:
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename="tokenizer.model",
                    token=self.config.hf_token,
                )
                shutil.copy2(downloaded, dest_spm)
                print(f"✓ Downloaded and installed tokenizer.model from {repo_id}")
                return dest_spm
            except Exception as e:
                print(f"Notice: Failed to download tokenizer.model from {repo_id}: {e}")

        raise FileNotFoundError(
            "CRITICAL: tokenizer.model is required for Gemma-3 GGUF conversion but could not be located or downloaded."
        )

    def convert_to_f16_gguf(
        self,
        merged_dir: Optional[Path | str] = None,
        output_gguf_path: Optional[Path | str] = None,
    ) -> Path:
        """Runs convert_hf_to_gguf.py to produce an FP16 GGUF file."""
        merged_path = Path(merged_dir or self.config.merged_dir).resolve()
        converter = self.find_llama_cpp_converter()
        gguf_dir = self.config.gguf_dir
        gguf_dir.mkdir(parents=True, exist_ok=True)

        out_f16 = Path(
            output_gguf_path or (gguf_dir / f"{self.config.run_name}-f16.gguf")
        ).resolve()

        # Create clean conversion workspace
        conversion_dir = gguf_dir / "hf_conversion_workspace"
        if conversion_dir.exists():
            shutil.rmtree(conversion_dir)
        conversion_dir.mkdir(parents=True, exist_ok=True)

        print(f"Preparing conversion workspace at {conversion_dir}...")
        for item in merged_path.iterdir():
            if item.is_file():
                shutil.copy2(item, conversion_dir / item.name)

        # Gemma 3 uses tokenizer.json (BPE). Only fallback to sentencepiece tokenizer.model if tokenizer.json is absent.
        if not (conversion_dir / "tokenizer.json").exists():
            self.ensure_sentencepiece_tokenizer(conversion_dir, self.config.base_model)
        else:
            print("✓ Found native tokenizer.json, using HuggingFace BPE tokenizer for GGUF conversion.")

        # Build conversion command
        cmd = [
            sys.executable,
            str(converter),
            str(conversion_dir),
            "--outfile",
            str(out_f16),
            "--outtype",
            "f16",
        ]

        print("=" * 80)
        print("RUNNING GGUF CONVERSION")
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
                f"llama.cpp GGUF conversion failed with exit code {result.returncode}.\nOutput:\n{result.stdout}"
            )

        if not out_f16.exists():
            raise FileNotFoundError(f"GGUF conversion completed but {out_f16} was not created.")

        f16_size_mb = round(out_f16.stat().st_size / (1024**2), 2)
        print(f"✓ FP16 GGUF created successfully: {out_f16} ({f16_size_mb} MB)")
        return out_f16
