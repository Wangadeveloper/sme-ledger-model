"""Tests for example formatting and label-masked tokenization."""

import pytest
from unittest.mock import MagicMock
from src.data.formatting import ExampleFormatter


class DummyTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<user>{messages[0]['content']}</user><assistant>"

    def __call__(self, text, add_special_tokens=False):
        # Fake tokenization: split into chars
        tokens = [ord(c) % 100 + 2 for c in text]
        return {"input_ids": tokens}


def test_formatting_and_label_masking():
    tok = DummyTokenizer()
    formatter = ExampleFormatter(tok, max_length=128)

    record = {
        "messages": [
            {"role": "user", "content": "You received Ksh 100"},
            {"role": "assistant", "content": '{"amount": 100}'},
        ]
    }

    formatted = formatter.format_chat_record(record)
    assert "<user>You received Ksh 100</user><assistant>" in formatted["prompt"]
    assert formatted["target"] == '{"amount": 100}'

    tokenized = formatter.tokenize_example(formatted)
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]

    # Check that prompt portion is masked with -100
    prompt_len = len(tok(formatted["prompt"])["input_ids"])
    assert labels[:prompt_len] == [-100] * prompt_len
    # Check that assistant target has valid labels
    assert labels[prompt_len:] == input_ids[prompt_len:]
    # Check EOS at end of target
    assert labels[-1] == tok.eos_token_id
