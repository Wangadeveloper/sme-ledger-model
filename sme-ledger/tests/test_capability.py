"""Tests for capability dataset handling, mixture creation, and evaluation."""

import pytest
from pathlib import Path
from src.data.capability import CapabilityDatasetManager
from src.evaluation.capability_eval import CapabilityEvaluator

CSV_PATH = Path("data/sme_ledger_v2_dataset/capability_examples.csv")


def test_capability_loading_and_categories():
    mgr = CapabilityDatasetManager(CSV_PATH)
    raw = mgr.load_and_validate()
    assert len(raw) == 31

    train_cap, held_out_cap = mgr.create_stratified_split(held_out_per_category=1)
    # Check that held out has exactly 10 questions (1 per category)
    assert len(held_out_cap) == 10
    assert len(train_cap) == 21

    # Check zero leakage between capability train and held out
    train_questions = {c["messages"][0]["content"] for c in train_cap}
    held_out_questions = {c["messages"][0]["content"] for c in held_out_cap}
    assert len(train_questions.intersection(held_out_questions)) == 0


def test_capability_mixture():
    mgr = CapabilityDatasetManager(CSV_PATH)
    train_cap, held_out_cap = mgr.create_stratified_split()
    dummy_tx = [{"messages": [{"role": "user", "content": f"SMS {i}"}, {"role": "assistant", "content": "{}"}]} for i in range(100)]

    mixed = mgr.build_training_mixture(dummy_tx, train_cap, target_ratio=0.1)
    assert len(mixed) > len(dummy_tx)
    cap_count = sum(1 for item in mixed if item.get("source_kind") == "capability")
    assert cap_count > 0


def test_capability_evaluation_scoring():
    held_out = [
        {
            "messages": [
                {"role": "user", "content": "Can you work offline?"},
                {"role": "assistant", "content": "SME-Ledger works offline on-device."}
            ],
            "category": "offline"
        }
    ]
    evaluator = CapabilityEvaluator(held_out)

    good_resp = "Yes, SME-Ledger is designed for local on-device processing and works completely offline without the internet."
    res = evaluator.evaluate_response("Can you work offline?", good_resp, held_out[0]["messages"][1]["content"], "offline")
    assert res["passed"] is True
    assert res["keyword_hit"] is True

    bad_resp = "{}"
    res_bad = evaluator.evaluate_response("Can you work offline?", bad_resp, held_out[0]["messages"][1]["content"], "offline")
    assert res_bad["passed"] is False
