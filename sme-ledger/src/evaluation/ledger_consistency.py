"""Ledger running balance reconciliation and anomaly detection for SME-Ledger V2."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import pandas as pd


class LedgerConsistencyEvaluator:
    """Evaluates mathematical consistency between extracted transaction amounts and reported balances."""

    @staticmethod
    def build_ledger_dataframe(extracted_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts extracted JSON transaction records into a typed DataFrame."""
        flat_records = []
        for rec in extracted_records:
            if isinstance(rec, list):
                flat_records.extend([r for r in rec if isinstance(r, dict)])
            elif isinstance(rec, dict):
                flat_records.append(rec)

        if not flat_records:
            return pd.DataFrame()

        df = pd.DataFrame(flat_records)
        for col in ["amount", "balance", "fee"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = None

        if "fee" in df.columns:
            df["fee"] = df["fee"].fillna(0.0)

        # Parse date and time for sequential ordering if available
        if "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
            df = df.sort_values("datetime", na_position="last").reset_index(drop=True)

        return df

    def evaluate_ledger_consistency(
        self, ledger: pd.DataFrame, tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """Calculates balance consistency across consecutive transactions with known balances."""
        if ledger.empty or len(ledger) < 2:
            return {
                "total_transactions": len(ledger),
                "reconcilable_transitions": 0,
                "consistent_transitions": 0,
                "inconsistent_transactions": 0,
                "balance_consistency_rate": 1.0 if len(ledger) <= 1 else 0.0,
                "financial_summary": {
                    "total_income": 0.0,
                    "total_expense": 0.0,
                    "net_cash_flow": 0.0,
                },
            }

        reconcilable = 0
        consistent = 0
        inconsistencies = []

        for i in range(1, len(ledger)):
            prev = ledger.iloc[i - 1]
            curr = ledger.iloc[i]

            prev_bal = prev.get("balance")
            curr_bal = curr.get("balance")
            amount = curr.get("amount")
            fee = curr.get("fee") or 0.0
            tx_type = str(curr.get("type") or "").strip().lower()

            if pd.isna(prev_bal) or pd.isna(curr_bal) or pd.isna(amount):
                continue

            reconcilable += 1

            if tx_type == "income":
                expected_bal = prev_bal + amount - fee
            elif tx_type == "expense":
                expected_bal = prev_bal - amount - fee
            else:
                expected_bal = prev_bal

            diff = abs(curr_bal - expected_bal)
            if diff <= tolerance:
                consistent += 1
            else:
                inconsistencies.append({
                    "index": i,
                    "transaction_id": curr.get("transaction_id"),
                    "prev_balance": float(prev_bal),
                    "amount": float(amount),
                    "fee": float(fee),
                    "type": tx_type,
                    "expected_balance": round(float(expected_bal), 2),
                    "actual_balance": float(curr_bal),
                    "discrepancy": round(float(diff), 2),
                })

        # Cash flow summary
        income_sum = float(ledger.loc[ledger["type"].astype(str).str.lower() == "income", "amount"].sum())
        expense_sum = float(ledger.loc[ledger["type"].astype(str).str.lower() == "expense", "amount"].sum())
        fee_sum = float(ledger["fee"].sum())

        return {
            "total_transactions": len(ledger),
            "reconcilable_transitions": reconcilable,
            "consistent_transitions": consistent,
            "inconsistent_transactions": len(inconsistencies),
            "balance_consistency_rate": round(consistent / max(reconcilable, 1), 4) if reconcilable > 0 else 1.0,
            "financial_summary": {
                "total_income": round(income_sum, 2),
                "total_expense": round(expense_sum, 2),
                "total_fees": round(fee_sum, 2),
                "net_cash_flow": round(income_sum - expense_sum - fee_sum, 2),
            },
            "sample_inconsistencies": inconsistencies[:5],
        }
