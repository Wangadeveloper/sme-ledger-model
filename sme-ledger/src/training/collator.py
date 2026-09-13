"""Causal language model data collator with dynamic padding and label masking."""

from __future__ import annotations

import torch
from typing import Any, Dict, List
from transformers import PreTrainedTokenizerBase


class CausalLMDataCollator:
    """Collator that pads batches dynamically and masks padding in the loss."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids = []
        attention_masks = []
        labels = []

        for feature in features:
            pad = max_len - len(feature["input_ids"])

            input_ids.append(
                feature["input_ids"] + [self.tokenizer.pad_token_id] * pad
            )
            attention_masks.append(
                feature["attention_mask"] + [0] * pad
            )
            labels.append(
                feature["labels"] + [-100] * pad
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
