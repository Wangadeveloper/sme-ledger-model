"""Tests for JSON extraction parser and canonicalization."""

import pytest
from src.evaluation.parser import extract_json_value, canonical_value


def test_extract_json_direct():
    text = '{"transaction_id": "ABC123", "amount": 500.0}'
    parsed = extract_json_value(text)
    assert isinstance(parsed, dict)
    assert parsed["transaction_id"] == "ABC123"
    assert parsed["amount"] == 500.0


def test_extract_json_with_surrounding_text():
    text = 'Here is the extracted data:\n```json\n{"transaction_id": "XYZ789", "amount": 1200.0}\n```\nHope that helps!'
    parsed = extract_json_value(text)
    assert isinstance(parsed, dict)
    assert parsed["transaction_id"] == "XYZ789"
    assert parsed["amount"] == 1200.0


def test_extract_json_array():
    text = '[{"transaction_id": "T1", "amount": 50.0}, {"transaction_id": "T2", "amount": 75.0}]'
    parsed = extract_json_value(text)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[1]["transaction_id"] == "T2"


def test_extract_json_invalid():
    assert extract_json_value("No json here at all") is None
    assert extract_json_value("") is None
    assert extract_json_value("Broken { json : 123 ") is None


def test_canonical_value():
    d1 = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 2}
    assert canonical_value(d1) == canonical_value(d2)
