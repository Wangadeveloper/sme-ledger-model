"""Tests for ledger running balance arithmetic, cash flow summary, and anomaly detection."""

import pytest
import pandas as pd
from src.evaluation.ledger_consistency import LedgerConsistencyEvaluator


def test_ledger_consistency_arithmetic():
    evaluator = LedgerConsistencyEvaluator()
    records = [
        {"transaction_id": "T1", "amount": 1000.0, "balance": 5000.0, "type": "income", "fee": 0.0, "date": "2026-04-01", "time": "10:00"},
        {"transaction_id": "T2", "amount": 200.0, "balance": 4790.0, "type": "expense", "fee": 10.0, "date": "2026-04-01", "time": "11:00"},
        {"transaction_id": "T3", "amount": 500.0, "balance": 5290.0, "type": "income", "fee": 0.0, "date": "2026-04-01", "time": "12:00"},
    ]

    df = evaluator.build_ledger_dataframe(records)
    metrics = evaluator.evaluate_ledger_consistency(df)

    assert metrics["total_transactions"] == 3
    assert metrics["reconcilable_transitions"] == 2
    assert metrics["consistent_transitions"] == 2
    assert metrics["inconsistent_transactions"] == 0
    assert metrics["balance_consistency_rate"] == 1.0
    assert metrics["financial_summary"]["total_income"] == 1500.0
    assert metrics["financial_summary"]["total_expense"] == 200.0
    assert metrics["financial_summary"]["total_fees"] == 10.0
    assert metrics["financial_summary"]["net_cash_flow"] == 1290.0


def test_ledger_inconsistency_detection():
    evaluator = LedgerConsistencyEvaluator()
    # T2 has an incorrect balance: 5000 - 200 = 4800, but reported balance is 3000!
    records = [
        {"transaction_id": "T1", "amount": 1000.0, "balance": 5000.0, "type": "income", "fee": 0.0, "date": "2026-04-01", "time": "10:00"},
        {"transaction_id": "T2", "amount": 200.0, "balance": 3000.0, "type": "expense", "fee": 0.0, "date": "2026-04-01", "time": "11:00"},
    ]

    df = evaluator.build_ledger_dataframe(records)
    metrics = evaluator.evaluate_ledger_consistency(df)

    assert metrics["consistent_transitions"] == 0
    assert metrics["inconsistent_transactions"] == 1
    assert metrics["balance_consistency_rate"] == 0.0
    assert len(metrics["sample_inconsistencies"]) == 1
    assert metrics["sample_inconsistencies"][0]["expected_balance"] == 4800.0
    assert metrics["sample_inconsistencies"][0]["actual_balance"] == 3000.0
