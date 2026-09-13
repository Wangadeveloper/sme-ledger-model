#!/usr/bin/env python3
"""Dataset validation script for SME-Ledger V2.

Usage:
    python scripts/validate_dataset.py --data-dir data/sme_ledger_v2_dataset --report-path reports/dataset_report.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validator import DatasetValidator


def main():
    parser = argparse.ArgumentParser(description="Validate SME-Ledger V2 dataset files and detect leakage.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sme_ledger_v2_dataset",
        help="Path to dataset directory containing jsonl and metadata files",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="reports/dataset_report.json",
        help="Destination path for dataset validation JSON report",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SME-LEDGER V2 DATASET VALIDATION")
    print(f"Data directory : {args.data_dir}")
    print(f"Report path    : {args.report_path}")
    print("=" * 80)

    try:
        validator = DatasetValidator(args.data_dir)
        report = validator.run_full_validation(report_path=args.report_path)

        print("\nSUMMARY OF SPLITS:")
        for name, data in report["splits"].items():
            print(f"  {name:12s}: {data['count']:,} records (avg user len: {data['avg_user_char_length']} chars)")

        print("\nCAPABILITY EXAMPLES:")
        cap_info = report["capability_examples"]
        print(f"  Total records: {cap_info['count']}")
        print(f"  Categories   : {list(cap_info['categories'].keys())}")

        print("\nLEAKAGE / DUPLICATE CHECK:")
        leak = report["leakage_and_duplicates"]
        for split, dup_count in leak["internal_duplicates"].items():
            print(f"  Internal duplicates in {split}: {dup_count}")

        for pair, info in leak["cross_split_leakage"].items():
            print(f"  Leakage {pair:22s}: exact={info['exact_overlap_count']}, near={info['near_overlap_count']}")

        print("\n✓ DATASET VALIDATION COMPLETE.")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Validation Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
