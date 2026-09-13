"""Tests for dataset loading, metadata validation, and cross-split leakage checks."""

import pytest
from pathlib import Path
from src.data.validator import DatasetValidator

DATA_DIR = Path("data/sme_ledger_v2_dataset")


def test_metadata_loading():
    validator = DatasetValidator(DATA_DIR)
    meta = validator.load_metadata()
    assert meta["name"] == "SME-Ledger V2 Synthetic Financial Intelligence Dataset"
    assert "send_money" in validator.supported_domains
    assert "transaction_id" in validator.schema_fields
    assert len(validator.supported_domains) == 16


def test_jsonl_splits_exist_and_non_empty():
    validator = DatasetValidator(DATA_DIR)
    for split in ["train.jsonl", "validation.jsonl", "test.jsonl", "benchmark.jsonl"]:
        path = DATA_DIR / split
        assert path.exists(), f"Missing split {split}"
        records = validator.load_jsonl(path)
        assert len(records) > 0, f"Split {split} is empty"


def test_no_cross_split_leakage():
    validator = DatasetValidator(DATA_DIR)
    splits = {
        "train": validator.load_jsonl(DATA_DIR / "train.jsonl"),
        "validation": validator.load_jsonl(DATA_DIR / "validation.jsonl"),
        "test": validator.load_jsonl(DATA_DIR / "test.jsonl"),
        "benchmark": validator.load_jsonl(DATA_DIR / "benchmark.jsonl"),
    }
    leakage = validator.check_leakage_and_duplicates(splits)
    for pair, info in leakage["cross_split_leakage"].items():
        assert info["exact_overlap_count"] == 0, f"Detected exact leakage in {pair}"
