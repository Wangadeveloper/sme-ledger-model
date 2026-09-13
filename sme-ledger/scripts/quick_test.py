#!/usr/bin/env python3
"""Quick interactive test: fire new prompts at the SME-Ledger V2 GGUF model."""

import json
import subprocess
from pathlib import Path

GGUF_MODEL = Path("/home/shadeform/sme-ledger/models/gguf/sme-ledger-v2-Q4_K_M.gguf")
LLAMA_CLI  = Path("/home/shadeform/llama.cpp/build/bin/llama-cli")

# ── New test prompts (never seen in training) ────────────────────────────────
TEST_PROMPTS = [
    # 1. Simple till payment
    {
        "label": "Till payment – groceries",
        "text": (
            "BZXPQM1234 Confirmed. Ksh4,500.00 paid to QUICKMART SUPERMARKET "
            "via Buy Goods and Services 543210. "
            "Date 12/9/2026 at 14:22. "
            "New M-PESA balance is Ksh21,350.75."
        ),
    },
    # 2. Send money (peer-to-peer)
    {
        "label": "Send money – friend",
        "text": (
            "TXN9087HJKL Confirmed. You have sent Ksh2,000.00 to JAMES MWANGI 0712345678 "
            "on 5/9/2026 at 09:15. "
            "New M-PESA balance is Ksh15,800.00. Transaction cost, Ksh29.00."
        ),
    },
    # 3. Receive money
    {
        "label": "Receive money",
        "text": (
            "QWERTY54321 Confirmed. You have received Ksh8,750.00 from ALICE NJERI 0798765432 "
            "on 11/9/2026 at 16:45. "
            "New M-PESA balance is Ksh30,100.50."
        ),
    },
    # 4. PayBill – rent
    {
        "label": "PayBill – rent",
        "text": (
            "LMNOP67890 Confirmed. Ksh35,000.00 paid to KENYA REAL ESTATE AGENCY "
            "via PayBill 111222. Account Number RENT-OCT-2026. "
            "Date 1/9/2026 at 08:00. "
            "New M-PESA balance is Ksh5,600.00. Transaction cost, Ksh105.00."
        ),
    },
    # 5. Withdraw from ATM (agent withdrawal)
    {
        "label": "ATM / Agent withdrawal",
        "text": (
            "WDRL334455 Confirmed. You have withdrawn Ksh10,000.00 from agent JOHN KAMAU 0722111222 "
            "on 10/9/2026 at 11:30. "
            "New M-PESA balance is Ksh12,400.25. Transaction cost, Ksh35.00."
        ),
    },
    # 6. Airtime top-up (self)
    {
        "label": "Airtime purchase",
        "text": (
            "ARTRP998877 Confirmed. Ksh100.00 airtime purchased for 0722999888 "
            "on 9/9/2026 at 07:05. "
            "New M-PESA balance is Ksh22,400.00."
        ),
    },
    # 7. Large paybill – electricity (KPLC)
    {
        "label": "PayBill – KPLC electricity",
        "text": (
            "ELEC556677 Confirmed. Ksh3,200.00 paid to KPLC PREPAID "
            "via PayBill 888880. Account Number 1234567890. "
            "Date 8/9/2026 at 19:22. "
            "New M-PESA balance is Ksh18,900.00. Transaction cost, Ksh0.00."
        ),
    },
    # 8. Edge case – zero balance after payment
    {
        "label": "Edge: zero balance after payment",
        "text": (
            "EDGE000111 Confirmed. Ksh500.00 paid to MAMA MBOGA STALL "
            "via Buy Goods and Services 654321. "
            "Date 13/9/2026 at 07:00. "
            "New M-PESA balance is Ksh0.00."
        ),
    },
]
# ─────────────────────────────────────────────────────────────────────────────


def run_prompt(text: str) -> str:
    prompt = (
        f"<start_of_turn>user\n{text}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    cmd = [
        str(LLAMA_CLI),
        "-m", str(GGUF_MODEL),
        "-p", prompt,
        "-n", "300",
        "--temp", "0.0",
        "-t", "4",
        "--no-display-prompt",
        "--single-turn",
        "--simple-io",
        "--log-disable",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    raw = proc.stdout
    # Strip any leaking prompt echoes
    for marker in ["<start_of_turn>model", "[ Prompt:", "Exiting..."]:
        if marker in raw:
            raw = raw.split(marker)[-1 if marker == "<start_of_turn>model" else 0]
            if marker != "<start_of_turn>model":
                raw = raw.split(marker)[0]
    return raw.strip()


def try_parse(text: str):
    try:
        # strip markdown fences if present
        cleaned = text.strip().strip("```json").strip("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def main():
    print(f"\n{'='*72}")
    print(f"  SME-Ledger V2 — Quick Prompt Test")
    print(f"  Model: {GGUF_MODEL.name}")
    print(f"{'='*72}\n")

    results = []
    for i, item in enumerate(TEST_PROMPTS, 1):
        print(f"[{i}/{len(TEST_PROMPTS)}] {item['label']}")
        print(f"  INPUT : {item['text'][:100]}{'...' if len(item['text'])>100 else ''}")

        raw = run_prompt(item["text"])
        parsed = try_parse(raw)
        valid_json = parsed is not None

        print(f"  OUTPUT: {raw[:120]}{'...' if len(raw)>120 else ''}")
        print(f"  STATUS: {'✅ Valid JSON' if valid_json else '❌ Not valid JSON'}")
        if valid_json:
            print(f"  PARSED: {json.dumps(parsed, ensure_ascii=False)}")
        print()

        results.append({
            "label": item["label"],
            "input": item["text"],
            "raw_output": raw,
            "parsed": parsed,
            "valid_json": valid_json,
        })

    # Summary
    passed = sum(1 for r in results if r["valid_json"])
    print(f"{'='*72}")
    print(f"  JSON Validity: {passed}/{len(results)} ({passed/len(results):.0%})")
    print(f"{'='*72}\n")

    # Save results
    out_path = Path("/home/shadeform/sme-ledger/reports/quick_test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
