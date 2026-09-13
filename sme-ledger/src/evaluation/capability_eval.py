"""Capability question evaluation on held-out test questions for SME-Ledger V2."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional


class CapabilityEvaluator:
    """Evaluates model performance on capability, privacy, and assistant questions."""

    # Keywords associated with SME-Ledger capabilities
    CAPABILITY_KEYWORDS = {
        "offline": ["offline", "on-device", "local", "internet", "device", "cloud"],
        "privacy": ["private", "privacy", "sensitive", "transmission", "remote", "cloud", "security", "stored"],
        "mpesa": ["m-pesa", "mpesa", "till", "paybill", "pochi", "fuliza", "transactions", "message", "sms"],
        "capabilities": ["extract", "parse", "analyze", "balance", "amount", "fee", "date", "entity", "domain", "cash-flow", "spending"],
        "bank": ["bank", "transfer", "account", "received", "sent", "statements"],
        "identity": ["sme-ledger", "assistant", "financial", "ledger", "bookkeeping", "small business"],
        "fraud": ["inconsistencies", "balance", "reconcil", "mismatch", "fraud", "suspicious", "anomal"],
        "limits": ["cannot", "not", "does not", "read-only", "cannot execute", "passive", "view", "analyze"],
        "cashflow": ["cash flow", "cash-flow", "income", "expense", "summary", "spending", "running"],
        "unknown": ["missing", "null", "omit", "incomplete", "unspecified"],
    }

    def __init__(self, held_out_test_records: List[Dict[str, Any]]):
        self.test_records = held_out_test_records

    def evaluate_response(
        self, question: str, response: str, reference: str, category: str
    ) -> Dict[str, Any]:
        """Assesses answer quality for a single capability question."""
        resp_clean = response.strip().lower()
        non_empty = len(resp_clean) >= 15

        # Check keyword presence
        expected_keys = self.CAPABILITY_KEYWORDS.get(category.lower(), ["sme-ledger", "financial"])
        found_keywords = [kw for kw in expected_keys if kw in resp_clean]
        keyword_hit = len(found_keywords) > 0

        # Absence of raw JSON dumps when answering natural language questions
        not_raw_json = not (resp_clean.startswith("{") and resp_clean.endswith("}"))

        passed = non_empty and keyword_hit and not_raw_json

        return {
            "question": question,
            "category": category,
            "reference": reference,
            "response": response,
            "passed": passed,
            "non_empty": non_empty,
            "keyword_hit": keyword_hit,
            "found_keywords": found_keywords,
            "not_raw_json": not_raw_json,
        }

    def evaluate(
        self, generate_fn: Callable[[str], str]
    ) -> Dict[str, Any]:
        """Runs generation function across all held-out capability test questions."""
        results: List[Dict[str, Any]] = []

        for item in self.test_records:
            messages = item["messages"]
            question = messages[0]["content"]
            reference = messages[1]["content"]
            category = item.get("category", "general")

            pred_text = generate_fn(question)
            eval_res = self.evaluate_response(question, pred_text, reference, category)
            results.append(eval_res)

        total = len(results)
        passed_count = sum(1 for r in results if r["passed"])
        pass_rate = round(passed_count / max(total, 1), 4)

        by_cat = {}
        for r in results:
            c = r["category"]
            by_cat.setdefault(c, {"total": 0, "passed": 0})
            by_cat[c]["total"] += 1
            if r["passed"]:
                by_cat[c]["passed"] += 1

        for c, data in by_cat.items():
            data["pass_rate"] = round(data["passed"] / max(data["total"], 1), 4)

        return {
            "total_questions": total,
            "passed_count": passed_count,
            "capability_pass_rate": pass_rate,
            "category_breakdown": by_cat,
            "sample_transcripts": results,
        }
