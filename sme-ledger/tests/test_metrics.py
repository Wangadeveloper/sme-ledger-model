"""Tests for field metrics evaluation and domain breakdowns."""

import pytest
from src.evaluation.field_metrics import FieldMetricsCalculator


def test_field_metrics_perfect_match():
    calc = FieldMetricsCalculator()
    pred = {
        "transaction_id": "ABC123",
        "date": "2026-04-20",
        "time": "14:30",
        "type": "income",
        "domain": "receive_money",
        "entity": "Jane Doe",
        "amount": 1000.0,
        "fee": 0.0,
        "balance": 5500.0,
        "reference": None,
    }
    ref = dict(pred)

    matches = calc.compare_single_record(pred, ref)
    assert matches["complete_record"] is True
    assert matches["amount"] is True
    assert matches["balance"] is True


def test_field_metrics_numeric_tolerance():
    calc = FieldMetricsCalculator()
    assert calc.compare_numeric(100.004, 100.0) is True
    assert calc.compare_numeric(100.5, 100.0) is False
    assert calc.compare_numeric(None, None) is True
    assert calc.compare_numeric(0.0, None) is False


def test_evaluate_predictions_batch():
    calc = FieldMetricsCalculator()
    predictions = [
        {
            "raw_output": '{"transaction_id": "T1", "amount": 100.0, "type": "income", "domain": "till_payment"}',
            "reference_text": '{"transaction_id": "T1", "amount": 100.0, "type": "income", "domain": "till_payment"}',
            "parsed_prediction": {"transaction_id": "T1", "amount": 100.0, "type": "income", "domain": "till_payment"},
            "parsed_reference": {"transaction_id": "T1", "amount": 100.0, "type": "income", "domain": "till_payment"},
        },
        {
            "raw_output": '{"transaction_id": "T2", "amount": 50.0, "type": "expense", "domain": "pochi"}',
            "reference_text": '{"transaction_id": "T2", "amount": 60.0, "type": "expense", "domain": "pochi"}',
            "parsed_prediction": {"transaction_id": "T2", "amount": 50.0, "type": "expense", "domain": "pochi"},
            "parsed_reference": {"transaction_id": "T2", "amount": 60.0, "type": "expense", "domain": "pochi"},
        }
    ]

    metrics = calc.evaluate_predictions(predictions)
    assert metrics["json_validity_rate"] == 1.0
    assert metrics["field_accuracies"]["transaction_id_accuracy"] == 1.0
    assert metrics["field_accuracies"]["amount_accuracy"] == 0.5
    assert "till_payment" in metrics["domain_breakdown"]
    assert "pochi" in metrics["domain_breakdown"]
