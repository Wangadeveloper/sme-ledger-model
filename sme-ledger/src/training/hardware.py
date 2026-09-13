"""Hardware detection, dtype selection, parameter validation, and memory utilities."""

from __future__ import annotations

import gc
import os
import psutil
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class HardwareInfo:
    """System hardware capabilities and selected training precision."""

    device_type: str
    device_name: str
    cuda_available: bool
    bf16_supported: bool
    fp16_supported: bool
    vram_total_gb: float
    cpu_count: int
    ram_total_gb: float
    selected_dtype: torch.dtype

    @classmethod
    def detect(cls, prefer_bf16: bool = False) -> HardwareInfo:
        cuda_avail = torch.cuda.is_available()
        device_type = "cuda" if cuda_avail else "cpu"
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
        bf16_supp = torch.cuda.is_bf16_supported() if cuda_avail else False
        fp16_supp = cuda_avail

        vram_gb = 0.0
        if cuda_avail:
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)

        cpu_count = os.cpu_count() or 1
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)

        # Precision selection: On CPU always use float32 to avoid instability.
        # On CUDA, follow notebook reference float32 unless prefer_bf16 is True.
        if not cuda_avail:
            selected_dtype = torch.float32
        elif prefer_bf16 and bf16_supp:
            selected_dtype = torch.bfloat16
        else:
            selected_dtype = torch.float32

        return cls(
            device_type=device_type,
            device_name=device_name,
            cuda_available=cuda_avail,
            bf16_supported=bf16_supp,
            fp16_supported=fp16_supp,
            vram_total_gb=vram_gb,
            cpu_count=cpu_count,
            ram_total_gb=ram_gb,
            selected_dtype=selected_dtype,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "cuda_available": self.cuda_available,
            "bf16_supported": self.bf16_supported,
            "fp16_supported": self.fp16_supported,
            "vram_total_gb": self.vram_total_gb,
            "cpu_count": self.cpu_count,
            "ram_total_gb": self.ram_total_gb,
            "selected_dtype": str(self.selected_dtype),
        }


def cleanup_memory() -> None:
    """Performs garbage collection and clears CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_memory(label: str = "") -> None:
    """Prints CPU RSS and GPU allocated memory."""
    rss_gb = round(psutil.Process(os.getpid()).memory_info().rss / (1024**3), 3)
    print(f"\n[MEMORY - {label}] CPU RSS: {rss_gb} GB", end="")
    if torch.cuda.is_available():
        alloc_gb = round(torch.cuda.memory_allocated() / (1024**3), 3)
        res_gb = round(torch.cuda.memory_reserved() / (1024**3), 3)
        print(f" | GPU Allocated: {alloc_gb} GB | GPU Reserved: {res_gb} GB", end="")
    print()


def check_model_parameters(model: torch.nn.Module, name: str = "model") -> None:
    """Verifies that all tensors in the model are finite (no NaN or Inf values)."""
    bad: List[str] = []
    total = 0

    for param_name, param in model.named_parameters():
        total += param.numel()
        if not torch.isfinite(param).all():
            bad.append(param_name)

    if bad:
        print(f"❌ {name} contains NaN/Inf parameters! Bad tensors: {bad[:10]}")
        raise RuntimeError(f"{name} contains NaN or Inf parameters.")

    print(f"✓ {name}: checked {total:,} parameters; all finite.")
