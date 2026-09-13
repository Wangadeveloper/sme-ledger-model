#!/usr/bin/env python3
"""Capability check: ask the model who it is, what it does, and its limits.
Tests fresh/rephrased questions across all trained capability categories."""

import json
import subprocess
from pathlib import Path

GGUF_MODEL = Path("/home/shadeform/sme-ledger/models/gguf/sme-ledger-v2-Q4_K_M.gguf")
LLAMA_CLI  = Path("/home/shadeform/llama.cpp/build/bin/llama-cli")

# ── New capability prompts (fresh phrasings, never in training CSV) ──────────
CAPABILITY_PROMPTS = [
    # identity
    {"category": "identity",      "q": "Hey, introduce yourself — what are you?"},
    {"category": "identity",      "q": "Tell me a bit about what you are built to do."},

    # capabilities
    {"category": "capabilities",  "q": "What kinds of transactions can you handle?"},
    {"category": "capabilities",  "q": "List everything you are able to extract from a message."},

    # mpesa
    {"category": "mpesa",         "q": "I just got an M-PESA confirmation SMS. Can you read it?"},
    {"category": "mpesa",         "q": "Do you support Fuliza and Pochi la Biashara?"},

    # bank
    {"category": "bank",          "q": "My Equity Bank sent me an SMS. Can you process it?"},
    {"category": "bank",          "q": "Do you only work with M-PESA or also bank messages?"},

    # offline / privacy
    {"category": "offline",       "q": "I have no internet connection. Can you still work?"},
    {"category": "privacy",       "q": "Does my transaction data leave my phone?"},
    {"category": "privacy",       "q": "Is it safe to give you my M-PESA messages?"},

    # limits
    {"category": "limits",        "q": "Can you transfer money to someone on my behalf?"},
    {"category": "limits",        "q": "If I ask you to pay a bill, will you do it?"},

    # unknown / hallucination guard
    {"category": "unknown",       "q": "What if the SMS is missing the transaction ID?"},
    {"category": "unknown",       "q": "Will you fill in missing amounts with a guess?"},

    # cashflow
    {"category": "cashflow",      "q": "Can you show me a summary of my spending this month?"},
    {"category": "cashflow",      "q": "How do you help with income and expense tracking?"},

    # fraud detection
    {"category": "fraud",         "q": "Can you tell me if a transaction looks suspicious?"},

    # out-of-scope / robustness
    {"category": "out_of_scope",  "q": "What is the weather in Nairobi today?"},
    {"category": "out_of_scope",  "q": "Write me a poem about bookkeeping."},
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
        "-n", "250",
        "--temp", "0.0",
        "-t", "4",
        "--no-display-prompt",
        "--single-turn",
        "--simple-io",
        "--log-disable",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    raw = proc.stdout
    for marker in ["<start_of_turn>model", "Exiting...", "[ Prompt:"]:
        if marker in raw:
            if marker == "<start_of_turn>model":
                raw = raw.split(marker)[-1]
            else:
                raw = raw.split(marker)[0]
    return raw.strip()


def grade(answer: str, category: str) -> str:
    """Simple heuristic grading for display."""
    ans_lower = answer.lower()

    # out-of-scope: model should decline / redirect
    if category == "out_of_scope":
        decline_signals = ["not able", "cannot", "can't", "not designed", "not built",
                           "financial", "transaction", "i am sme", "my purpose", "i don't"]
        return "✅ Declined / redirected" if any(s in ans_lower for s in decline_signals) else "⚠️  Answered (may be hallucinating)"

    # identity: should say SME-Ledger
    if category == "identity":
        return "✅" if "sme-ledger" in ans_lower or "sme ledger" in ans_lower else "⚠️  No self-identification"

    # limits: should NOT claim it can do payments/transfers
    if category == "limits":
        bad = ["yes", "sure", "i can transfer", "i can pay", "i will", "i'll"]
        good = ["cannot", "can't", "not able", "do not", "don't", "analyze", "i do not initiate"]
        if any(b in ans_lower for b in bad):
            return "❌ Claims capability it shouldn't have"
        if any(g in ans_lower for g in good):
            return "✅ Correctly declined"
        return "⚠️  Unclear"

    # unknown / hallucination guard
    if category == "unknown":
        safe = ["null", "unavailable", "not invent", "missing", "absent", "cannot guess", "do not invent", "return null"]
        return "✅ Won't hallucinate" if any(s in ans_lower for s in safe) else "⚠️  May hallucinate"

    # privacy / offline
    if category in ("privacy", "offline"):
        ok = ["local", "on-device", "offline", "device", "not upload", "not transmit", "minimiz"]
        return "✅" if any(s in ans_lower for s in ok) else "⚠️  Unclear"

    # everything else: just check it's non-empty and reasonable length
    return "✅" if len(answer) > 20 else "⚠️  Very short / no answer"


def main():
    print(f"\n{'='*72}")
    print(f"  SME-Ledger V2 — Capability Check")
    print(f"  Model : {GGUF_MODEL.name}")
    print(f"  Prompts: {len(CAPABILITY_PROMPTS)}")
    print(f"{'='*72}\n")

    results = []
    for i, item in enumerate(CAPABILITY_PROMPTS, 1):
        cat = item["category"].upper()
        q   = item["q"]
        print(f"[{i:02d}/{len(CAPABILITY_PROMPTS)}] [{cat}]")
        print(f"  Q: {q}")
        answer = run_prompt(q)
        verdict = grade(answer, item["category"])
        print(f"  A: {answer}")
        print(f"  → {verdict}")
        print()
        results.append({"category": item["category"], "question": q, "answer": answer, "verdict": verdict})

    # Summary by category
    print(f"{'='*72}")
    print("  SUMMARY BY CATEGORY")
    print(f"{'='*72}")
    cats = sorted(set(r["category"] for r in results))
    for cat in cats:
        cat_results = [r for r in results if r["category"] == cat]
        good = sum(1 for r in cat_results if r["verdict"].startswith("✅"))
        print(f"  {cat:<16} {good}/{len(cat_results)} ✅")

    total_good = sum(1 for r in results if r["verdict"].startswith("✅"))
    print(f"\n  OVERALL : {total_good}/{len(results)} ({total_good/len(results):.0%})")
    print(f"{'='*72}\n")

    # Save
    out = Path("/home/shadeform/sme-ledger/reports/capability_check_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {out}")


if __name__ == "__main__":
    main()
