"""Configuration dataclass for the SME-Ledger V2 training pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainingPipelineConfig:
    """End-to-end pipeline configuration."""

    # Model identifiers
    base_model: str = "google/gemma-3-270m-it"
    fallback_base_model: str = "Huihui-ai/Huihui-gemma-3-270m-it-abliterated"
    hf_token: Optional[str] = None

    # Paths
    project_root: str = "."
    data_dir: str = "data/sme_ledger_v2_dataset"
    output_dir: str = "models"
    reports_dir: str = "reports"
    run_name: str = "sme-ledger-v2"
    llama_cpp_dir: str = "/home/shadeform/llama.cpp"

    # Reproducibility
    seed: int = 20260910

    # LoRA Hyperparameters (from notebook reference)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Optimization Hyperparameters
    learning_rate: float = 1e-4
    num_epochs: int = 5
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 768
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Generation & Evaluation
    max_new_tokens: int = 300
    eval_limit: int = 100
    benchmark_eval_limit: int = 50
    capability_ratio: float = 0.04

    # Quantization
    quantization_type: str = "Q4_K_M"

    def __post_init__(self):
        # Allow environment variable override for HF_TOKEN
        if not self.hf_token:
            self.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        # Resolve paths
        self.project_root = str(Path(self.project_root).resolve())
        self.data_dir = str(Path(self.data_dir).resolve())
        self.output_dir = str(Path(self.output_dir).resolve())
        self.reports_dir = str(Path(self.reports_dir).resolve())

    @property
    def checkpoints_dir(self) -> Path:
        return Path(self.output_dir) / "checkpoints"

    @property
    def lora_dir(self) -> Path:
        return Path(self.output_dir) / "lora"

    @property
    def merged_dir(self) -> Path:
        return Path(self.output_dir) / "merged"

    @property
    def gguf_dir(self) -> Path:
        return Path(self.output_dir) / "gguf"

    @property
    def manifests_dir(self) -> Path:
        return Path(self.output_dir) / "manifests"

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to a serializable dictionary, masking sensitive tokens."""
        d = asdict(self)
        if d.get("hf_token"):
            d["hf_token"] = "***MASKED***"
        return d

    def save_json(self, path: Path | str) -> None:
        """Saves configuration to JSON."""
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> TrainingPipelineConfig:
        """Loads configuration from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
