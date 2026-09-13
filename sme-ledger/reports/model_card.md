# SME-Ledger V2 Model Card

## Overview

SME-Ledger V2 is an offline-first, on-device financial intelligence model designed for Small and Medium Enterprises (SMEs) in East Africa. It extracts structured accounting records from raw mobile money (M-PESA) and banking SMS notifications, reconciles running cash balances, and answers capability and privacy questions without leaking data to cloud endpoints.

## Model Details

- **Base Model**: `google/gemma-3-270m-it`
- **Fine-Tuning Method**: LoRA (PEFT)
- **LoRA Parameters**: Rank=8, Alpha=16, Dropout=0.05
- **Quantization**: `Q4_K_M` (GGUF format)
- **Final GGUF File**: `sme-ledger-v2-Q4_K_M.gguf` (249.21 MB)
- **Deployment Target**: Edge mobile / low-resource on-device runtime (`llama.cpp`)

## Dataset

- **Train Count**: 15,624 examples
- **Validation Count**: 1,953 examples
- **Test Count**: 1,954 examples
- **Benchmark Count**: 2,000 examples (strictly isolated)
- **Capability Instruction Queries**: 31 examples across 10 categories

## Training Hyperparameters

- **Epochs**: 5
- **Learning Rate**: 0.0001
- **Batch Size**: 1 (effective: 8)
- **Sequence Length**: 768
- **Random Seed**: 20260910

## Evaluation Performance

| Task / Metric | Result |
| :--- | :--- |
| **JSON Validity Rate** | `98.00%` |
| **Complete Record Accuracy** | `42.00%` |
| **Isolated Benchmark Accuracy** | `6.00%` |
| **Held-Out Capability Pass Rate** | `70.00%` |
| **Ledger Balance Consistency** | `0.00%` |

## Intended Use & Safety

- **Primary Use**: Local, offline parsing of M-PESA and banking SMS receipts into double-entry accounting schemas.
- **Privacy**: Zero external network requests during inference; sensitive financial data remains strictly on-device.
- **Limitations**: Does not execute financial transactions; operates in read-only analytical mode.
