# SME-Ledger V2 Evaluation Report

**Model Path**: `/home/shadeform/sme-ledger/models/gguf/sme-ledger-v2-Q4_K_M.gguf`  
**Model Type**: `GGUF`  
**Evaluated Examples**: 100 test records

## 1. Test Set Extraction Performance

| Metric | Score |
| :--- | :--- |
| **JSON Validity Rate** | `98.00%` |
| **Exact Match Rate** | `42.00%` |
| **Complete Record Accuracy** | `42.00%` |

### Field-Level Accuracies

| Field | Accuracy |
| :--- | :--- |
| `amount` | `90.57%` |
| `balance` | `90.57%` |
| `date` | `69.81%` |
| `domain` | `85.85%` |
| `entity` | `89.62%` |
| `fee` | `90.57%` |
| `reference` | `89.62%` |
| `time` | `45.28%` |
| `transaction_id` | `86.79%` |
| `type` | `90.57%` |

### Domain Breakdown

| Domain | Total | Correct | Accuracy |
| :--- | :--- | :--- | :--- |
| `airtime` | 12 | 0 | `0.00%` |
| `bank_to_mpesa` | 3 | 0 | `0.00%` |
| `bank_transfer_received` | 9 | 5 | `55.56%` |
| `bank_transfer_sent` | 6 | 3 | `50.00%` |
| `cash_withdrawal` | 6 | 6 | `100.00%` |
| `failed_transaction` | 4 | 0 | `0.00%` |
| `fuliza_drawdown` | 3 | 0 | `0.00%` |
| `fuliza_fee` | 8 | 0 | `0.00%` |
| `fuliza_repayment` | 6 | 0 | `0.00%` |
| `mpesa_to_bank` | 10 | 7 | `70.00%` |
| `paybill` | 10 | 5 | `50.00%` |
| `pochi` | 5 | 2 | `40.00%` |
| `receive_money` | 4 | 3 | `75.00%` |
| `reversal` | 4 | 0 | `0.00%` |
| `send_money` | 8 | 5 | `62.50%` |
| `till_payment` | 8 | 6 | `75.00%` |

## 2. Isolated Unseen Benchmark Performance

| Metric | Score |
| :--- | :--- |
| **Evaluated Records** | 50 |
| **JSON Validity Rate** | `96.00%` |
| **Exact Match Rate** | `6.00%` |
| **Complete Record Accuracy** | `6.00%` |

## 3. Held-Out Capability & Privacy Evaluation

**Overall Capability Pass Rate**: `70.00%` (7/10 questions)

| Category | Questions | Passed | Pass Rate |
| :--- | :--- | :--- | :--- |
| `bank` | 1 | 1 | `100.00%` |
| `capabilities` | 1 | 1 | `100.00%` |
| `cashflow` | 1 | 0 | `0.00%` |
| `fraud` | 1 | 1 | `100.00%` |
| `identity` | 1 | 1 | `100.00%` |
| `limits` | 1 | 1 | `100.00%` |
| `mpesa` | 1 | 1 | `100.00%` |
| `offline` | 1 | 1 | `100.00%` |
| `privacy` | 1 | 0 | `0.00%` |
| `unknown` | 1 | 0 | `0.00%` |

## 4. Ledger Running Balance Consistency

| Metric | Value |
| :--- | :--- |
| **Reconcilable Transitions** | 68 |
| **Balance Consistency Rate** | `0.00%` |
| **Detected Inconsistencies** | 68 |
| **Reconstructed Cash Flow** | Income: Ksh 636,631.40 \| Expenses: Ksh 990,754.22 \| Net: Ksh -355,610.15 |
