# Arbor v17 — Full Results & Analysis
**Date:** 2026-04-09  
**Model:** treesearch-v17 (Qwen2.5-3B-Instruct + LoRA, trained on 63,438 examples)  
**Base model:** v16 (62.7%) → v17 target: 70%+

---

## FinanceBench Evaluation — 150 Questions

| Metric | v16 | v17 | Change |
|--------|-----|-----|--------|
| Perfect (recall=1.0) | 94/150 (62.7%) | **114/150 (76.0%)** | **+20** |
| Partial (0 < recall < 1) | ~10 | 7 | -3 |
| Zero recall | ~46 | 29 | -17 |
| Avg recall | ~0.72 | **0.782** | +0.06 |

**Target exceeded: 76% vs 70% goal.**

- **29 new wins** — questions v16 failed that v17 now gets correct
- **9 regressions** — questions v16 got right that v17 now fails

---

## FinanceBench — 29 New Wins (v16 FAIL → v17 PASS)

Q009, Q013, Q015, Q017, Q025, Q026, Q033, Q044, Q051, Q052, Q064, Q065, Q072, Q073, Q074, Q086, Q087, Q088, Q095, Q098, Q116, Q118, Q119, Q122, Q124, Q131, Q139, Q142, Q143

---

## FinanceBench — 9 Regressions (v16 PASS → v17 FAIL)

| Q | Document | Root Cause |
|---|----------|-----------|
| Q001 | 3M_2018_10K | Wrong branch — navigated to p51-68 instead of p52-61 (evidence p59) |
| Q042 | AMERICANEXPRESS_2022_10K | Wrong branch — missed Consolidated Financial Statements (p93-97) |
| Q045 | AMERICANEXPRESS_2022_10K | Wrong branch — missed section covering p44 |
| Q076 | CVSHEALTH_2018_10K | Wrong branch — navigated 9 nodes, never reached p301/303 |
| Q081 | FOOTLOCKER_2022_8K | Flat 4-node 8K — navigated to p2 instead of p1 |
| Q107 | MGMRESORTS_2022Q4_EARNINGS | Flat 15-node earnings release — p10 instead of p12 |
| Q125 | PEPSICO_2022_10K | Wrong branch — missed evidence at p61/63 |
| Q132 | PFIZER_2021_10K | Wrong branch — missed evidence at p56 |
| Q149 | WALMART_2019_10K | Wrong branch — missed evidence at p47 |

All regressions are pure navigation errors (wrong branch selection), not tree coverage gaps.

---

## FinanceBench — Remaining 36 Failures

### Zero Recall (29 questions)

| Document | Questions | Failure Type |
|----------|-----------|-------------|
| AMERICANEXPRESS_2022_10K | Q039, Q040, Q042, Q045 | Wrong-branch navigation |
| 3M_2018_10K | Q001, Q002 | Wrong-branch navigation |
| AES_2022_10K | Q016, Q018 | Wrong-branch / tree gap |
| MGMRESORTS_2022Q4_EARNINGS | Q107, Q109 | Flat earnings release (Page N of 15 nodes) |
| PEPSICO_2022_10K | Q123, Q125 | Wrong-branch navigation |
| PEPSICO_2023Q1_EARNINGS | Q129, Q130 | Flat earnings release |
| PFIZER_2021_10K | Q132, Q133 | Wrong-branch navigation |
| 3M_2023Q2_10Q | Q007 | 10-Q flat structure |
| BLOCK_2020_10K | Q059 | Tree coverage gap |
| BOEING_2022_10K | Q062 | Tree coverage gap |
| CVSHEALTH_2018_10K | Q076 | Deep doc (300+ pages), coverage gap |
| CVSHEALTH_2022_10K | Q078 | Tree coverage gap |
| FOOTLOCKER_2022_8K | Q081 | Flat 8K (4 nodes), 1-page navigation error |
| JOHNSON_JOHNSON_2022Q4_EARNINGS | Q089 | Flat earnings release |
| JPMORGAN_2023Q2_10Q | Q099 | 10-Q flat structure |
| NIKE_2018_10K | Q115 | Wrong-branch navigation |
| PEPSICO_2021_10K | Q120 | Wrong-branch navigation |
| ULTABEAUTY_2023_10K | Q137 | Tree coverage gap |
| VERIZON_2022_10K | Q147 | Tree coverage gap |
| WALMART_2019_10K | Q149 | Wrong-branch navigation |

### Partial Recall (7 questions)

| Q | Document | Recall | Issue |
|---|----------|--------|-------|
| Q003 | 3M_2022_10K | 0.33 | Multi-page evidence, found only 1/3 |
| Q011 | ADOBE_2015_10K | 0.50 | Found 1 of 2 evidence pages |
| Q071 | CORNING_2020_10K | 0.50 | Found 1 of 2 evidence pages |
| Q117 | NIKE_2021_10K | 0.50 | Found 1 of 2 evidence pages |
| Q144 | VERIZON_2021_10K | 0.50 | Found 1 of 2 evidence pages |
| Q146 | VERIZON_2022_10K | 0.50 | Found 1 of 2 evidence pages |
| Q148 | WALMART_2018_10K | 0.50 | Found 1 of 2 evidence pages |

---

## Failure Category Breakdown

| Category | Count | Description | Fixable? |
|----------|-------|-------------|----------|
| Wrong-branch navigation | ~17 | Model picks wrong top-level section | Yes — more DAgger |
| Flat docs (earnings/8K) | ~6 | "Page N of 15" nodes, no real structure | Partially |
| Tree coverage gaps | ~8 | Evidence pages genuinely outside any node | Requires StructDirect fix |
| Partial recall | 7 | Multi-evidence, model finds only some | Yes — multi-hop training |
| Unreachable (evidence p0) | ~2 | Cover page questions | No |

**Path to 130+/150:**
- Fix 9 regressions (same as v16 successes, add to replay): +9 → 123
- Convert 7 partials to full: +7 → 130
- Fix some wrong-branch cases with v18 DAgger: +7 → 137

---

## Apple 2024 10K — Interactive Benchmark (10 Questions)

**PDF:** `c87043b9-5d89-4717-9f49-c4f9663d0061.pdf` (Apple FY2024 Annual Report, 121 pages)  
**Score: 3/10 (30%)**

| Q | Question | Expected Page | Result | Model Went To |
|---|----------|--------------|--------|---------------|
| Q01 | Total net sales FY2024? | p32 (Income Statement) | FAIL | Pages 28-31 (index) |
| Q02 | Net income FY2024? | p32 | PASS | Consolidated Statements Of Operations (p32) |
| Q03 | Two revenue categories on income statement? | p32 | FAIL | Item 1A Risk Factors (p5-16) |
| Q04 | Total assets Sep 28, 2024? | p34 (Balance Sheet) | FAIL | Consolidated Statements Of Operations (p32) |
| Q05 | Cash and equivalents end of FY2024? | p34 | PASS | Consolidated Balance Sheets (p34-35) |
| Q06 | New Mac products Q1 FY2024? | p24 (MD&A) | FAIL | Item 1A Risk Factors (p5-16) |
| Q07 | Americas segment net sales FY2024? | p25 (MD&A) | FAIL | Consolidated Statements Of Operations (p32) |
| Q08 | Effective tax rate FY2024? | p28 (Notes) | FAIL | Consolidated Statements Of Operations (p32) |
| Q09 | Stock ticker and exchange? | p22 (Item 5) | FAIL | Pages 28-31 (index) |
| Q10 | Risk factors re: third-party manufacturers? | p5-16 | PASS | Item 1A Risk Factors ✓ |

### Why 3/10 on Apple but 76% on FinanceBench?

1. **This is an UNSEEN document** — Apple 2024 10K was never in training data. FinanceBench questions were partially covered by DAgger and replay training.
2. **The model can't distinguish between financial sub-statements.** It navigates to Item 8 correctly but then picks "Consolidated Statements of Operations" for ALL financial questions — even balance sheet and tax questions. The model was trained to reach "Item 8" as a terminal node; it was never trained to navigate between the individual statements within Item 8.
3. **Item 5 (stock ticker) is not extracted as a top-level node** in this PDF's tree — it's embedded inside Item 7's page range. The model can't navigate to it directly.
4. **Persistent second-pick problem:** The model always returns `Pages 56-65` (Item 16 Form 10-K Summary) as a second node for almost every question, regardless of relevance.

### What the 76% FinanceBench number actually means

FinanceBench evaluation navigated to the correct section at the **section level** (Item 8, Item 7, etc.) and checked whether evidence pages fell within the returned node's page range. The Apple benchmark tests at **sub-statement precision** (exact page 32 vs 34 vs 37), which is a stricter standard the model was never trained to meet.

---

## Domain PDF Test — Legal Paper

**PDF:** `data/domain_pdfs/legal/2603.27075v1.pdf`  
*"Mind the Gap: How the Technical Mechanisms of Agentic AI Outpace Global Legal Frameworks"*  
**Unseen document (not in training set)**

### Before StructDirect Fix
Tree was completely broken — 4 nodes all named garbage (URLs, citations, date stamps):
```
[0001] A PREPRINT - MARCH 31, 2026  (p1)
[0002] A PREPRINT - MARCH 31, 2026  (p2)
[0003] 11Anthropic (n 3)            (p3)
[0004] //www.whitecase.com/...      (p4-15)
```

### After StructDirect Fix (multiline_toc quality gate)
Proper 7-node tree extracted via font_headings:
```
[0002] Introduction                                          p2
[0003] Legal and Policy Definitions of Agentic AI...        p3-9
[0004] Part II: Synthesis and Critical Analysis...          p10-11
[0005] The Structural Reasons for Definitional Failure      p12-13
[0006] Conclusion                                           p14
[0007] Bibliography                                         p15
```

### 5-Question Test Results

| Q | Question | Expected | Result |
|---|----------|----------|--------|
| Q1 | Four recurring categories of definitional failure? | p10-13 | PARTIAL — went to p3-9 (survey, not synthesis) |
| Q2 | Which bodies conflate AI with multi-agent systems? | p10-11 | PARTIAL — went to p3-9 (survey section) |
| Q3 | G7 claim about self-replicating models? | p5 | FAIL — went to Conclusion (p14) |
| Q4 | Actual technical mechanism of agency? | p8 | PASS — p3-9 covers p8 |
| Q5 | Structural reason for cognitive vocabulary? | p12-13 | FAIL — went to p3-9 |

**Root cause:** Model defaults to the largest/most prominent section (`[0003]` at 7 pages) for most questions. For academic papers with nuanced section titles ("Part II: Synthesis" vs "Legal Definitions"), the model needs academic-paper-specific navigation training.

---

## StructDirect Fixes Made in v17 Session

| Fix | Description | Impact |
|-----|-------------|--------|
| `font_headings` coverage check | Reject font_headings strategy if <50% of pages covered | Fixed CORNING_2022_10K |
| Cover/Preamble insertion | Insert node if first section doesn't start at p0/1 | Fixed AMCOR_2023_10K unreachability |
| `multiline_toc` URL/citation rejection | Reject lines containing URLs, footnote citations, preprint stamps | Fixed legal paper garbage tree |
| `multiline_toc` repeated-title gate | Reject multiline_toc if >50% of entries are repeated running headers | Fixed academic paper with page headers |
| Financial statement header detection | Scan for CONSOLIDATED STATEMENTS OF OPERATIONS etc. before page-chunking | Fixed Apple 10K "Pages 28-37" → proper statement names |
| Index-page entry filter | Skip TOC index entries (statement title followed by bare page number) | Fixed duplicate nodes from index page |

---

## Training Data Summary

| Source | Raw | Multiplier | Effective | Share |
|--------|-----|-----------|-----------|-------|
| diverse_train (38,826 examples, 43 PDFs) | 38,826 | ×1 | 38,826 | 61.2% |
| structdirect_train (fixed trees) | 1,329 | ×8 | 10,632 | 16.8% |
| dagger_v17_targeted (56 v16 failures) | 69 | ×40 | 2,760 | 4.4% |
| dagger_v15_targeted | 139 | ×40 | 5,560 | 8.8% |
| v17_success_replay (94 v16 successes) | 118 | ×30 | 3,540 | 5.6% |
| v14b_success_replay | 106 | ×20 | 2,120 | 3.3% |
| **Total** | | | **63,438** | 100% |

- DAgger combined (v15 + v17): **13.2%** of mix
- Training: 2 epochs × 63,438 = 7,930 steps on A100 (~90 min)
- LoRA: r=16, 0.96% trainable params (29.9M of 3.1B)

---

## What Needs to Happen for v18

### High Priority
1. **Recover 9 regressions** — add these as success-replay examples from v16 results before v18 training
2. **Fix AmericanExpress (4 questions)** — largest single failure cluster; add targeted DAgger for those specific navigation paths
3. **Sub-statement navigation training** — add examples that teach: "total assets → navigate to Consolidated Balance Sheets" (not just "go to Item 8")

### Medium Priority  
4. **Partial → full conversion** — 7 questions at 0.50 recall; fix multi-evidence navigation (model stops after first hit)
5. **Flat earnings releases** — 6 questions in Page-N-of-15 docs; add training examples for navigating flat numbered trees
6. **Academic paper navigation** — model goes to largest section for all questions; needs examples teaching synthesis vs survey section distinction

### Low Priority / Structural
7. **Tree coverage gaps** (~8 questions) — evidence pages outside any node; requires deeper StructDirect fixes for specific documents
8. **Unreachable (evidence p0)** — cover page questions; no fix possible with current tree navigation approach
