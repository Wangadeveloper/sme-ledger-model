"""Field-level and domain-level accuracy evaluation for SME-Ledger V2."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from src.evaluation.parser import extract_json_value, canonical_value


class FieldMetricsCalculator:
    """Calculates field-level, complete-record, and per-domain accuracy."""

    FIELD_KEYS = [
        "transaction_id",
        "date",
        "time",
        "type",
        "domain",
        "entity",
        "amount",
        "fee",
        "balance",
        "reference",
    ]

    @staticmethod
    def compare_numeric(pred_val: Any, ref_val: Any, tol: float = 1e-2) -> bool:
        """Compares numeric values with tolerance, handling None / null."""
        if pred_val is None and ref_val is None:
            return True
        if pred_val is None or ref_val is None:
            return False
        try:
            p_float = float(pred_val)
            r_float = float(ref_val)
            return abs(p_float - r_float) <= tol
        except (ValueError, TypeError):
            return False

    @staticmethod
    def compare_strings(pred_val: Any, ref_val: Any) -> bool:
        """Compares string fields case-insensitively with stripped whitespace."""
        if pred_val is None and ref_val is None:
            return True
        if pred_val is None or ref_val is None:
            return False
        return str(pred_val).strip().lower() == str(ref_val).strip().lower()

    def compare_single_record(
        self, pred_dict: Dict[str, Any], ref_dict: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Compares individual fields between predicted and reference record."""
        field_matches = {}

        # String fields
        for key in ["transaction_id", "date", "time", "type", "domain", "entity", "reference"]:
            field_matches[key] = self.compare_strings(pred_dict.get(key), ref_dict.get(key))

        # Numeric fields
        for key in ["amount", "fee", "balance"]:
            field_matches[key] = self.compare_numeric(pred_dict.get(key), ref_dict.get(key))

        # All fields match
        field_matches["complete_record"] = all(field_matches.values())
        return field_matches

    def evaluate_predictions(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluates a collection of prediction items against references.

        Each item in predictions should be:
        {
            "raw_output": str,
            "reference_text": str,
            "parsed_prediction": Optional[Any],
            "parsed_reference": Optional[Any],
        }
        """
        total = len(predictions)
        if total == 0:
            return {"total": 0}

        valid_json_count = 0
        exact_match_count = 0

        field_correct_counts = {k: 0 for k in self.FIELD_KEYS}
        field_total_counts = {k: 0 for k in self.FIELD_KEYS}
        complete_record_correct = 0
        evaluable_records = 0

        domain_correct_counts: Dict[str, int] = {}
        domain_total_counts: Dict[str, int] = {}

        for item in predictions:
            raw_out = item.get("raw_output", "")
            ref_text = item.get("reference_text", "")

            pred_val = item.get("parsed_prediction")
            if pred_val is None:
                pred_val = extract_json_value(raw_out)

            ref_val = item.get("parsed_reference")
            if ref_val is None:
                ref_val = extract_json_value(ref_text)

            is_valid_json = pred_val is not None
            if is_valid_json:
                valid_json_count += 1

            # Exact match check
            if pred_val is not None and ref_val is not None:
                if canonical_value(pred_val) == canonical_value(ref_val):
                    exact_match_count += 1
            elif raw_out.strip() == ref_text.strip():
                exact_match_count += 1

            # Field-level comparison
            if isinstance(ref_val, dict):
                evaluable_records += 1
                domain = str(ref_val.get("domain", "unknown")).lower()
                domain_total_counts[domain] = domain_total_counts.get(domain, 0) + 1

                if isinstance(pred_val, dict):
                    matches = self.compare_single_record(pred_val, ref_val)
                    for k in self.FIELD_KEYS:
                        field_total_counts[k] += 1
                        if matches[k]:
                            field_correct_counts[k] += 1

                    if matches["complete_record"]:
                        complete_record_correct += 1
                        domain_correct_counts[domain] = domain_correct_counts.get(domain, 0) + 1
                else:
                    for k in self.FIELD_KEYS:
                        field_total_counts[k] += 1

            elif isinstance(ref_val, list):
                # Multi-transaction list
                evaluable_records += 1
                if isinstance(pred_val, list) and len(pred_val) == len(ref_val):
                    all_sub_match = True
                    for sub_pred, sub_ref in zip(pred_val, ref_val):
                        if isinstance(sub_pred, dict) and isinstance(sub_ref, dict):
                            matches = self.compare_single_record(sub_pred, sub_ref)
                            d = str(sub_ref.get("domain", "unknown")).lower()
                            domain_total_counts[d] = domain_total_counts.get(d, 0) + 1
                            if matches["complete_record"]:
                                domain_correct_counts[d] = domain_correct_counts.get(d, 0) + 1
                            else:
                                all_sub_match = False
                            for k in self.FIELD_KEYS:
                                field_total_counts[k] += 1
                                if matches[k]:
                                    field_correct_counts[k] += 1
                        else:
                            all_sub_match = False
                    if all_sub_match:
                        complete_record_correct += 1
                else:
                    for sub_ref in ref_val:
                        if isinstance(sub_ref, dict):
                            d = str(sub_ref.get("domain", "unknown")).lower()
                            domain_total_counts[d] = domain_total_counts.get(d, 0) + 1
                            for k in self.FIELD_KEYS:
                                field_total_counts[k] += 1

        field_accuracies = {}
        for k in self.FIELD_KEYS:
            tot = field_total_counts[k]
            field_accuracies[f"{k}_accuracy"] = round(field_correct_counts[k] / max(tot, 1), 4)

        domain_accuracies = {}
        for d, count in domain_total_counts.items():
            corr = domain_correct_counts.get(d, 0)
            domain_accuracies[d] = {
                "total": count,
                "correct": corr,
                "accuracy": round(corr / max(count, 1), 4),
            }

        return {
            "total_examples": total,
            "evaluable_transaction_records": evaluable_records,
            "json_validity_rate": round(valid_json_count / total, 4),
            "exact_match_rate": round(exact_match_count / total, 4),
            "complete_record_accuracy": round(complete_record_correct / max(evaluable_records, 1), 4),
            "field_accuracies": field_accuracies,
            "domain_breakdown": domain_accuracies,
        }
