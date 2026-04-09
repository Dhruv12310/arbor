# Full Reconstruct — Core Problem Analysis

Now I have the full picture. This is the honest, deep analysis.

---
The Core Problem — Three Layers Deep

Layer 1: The Training Data is a Lie (Memorization, Not Learning)

38,826 training examples from 39 documents. That's ~1,000 examples per document on average. The model didn't learn to navigate — it memorized the layout of 39 specific PDFs.

- 33% finance examples, 67% academic — but FinanceBench is 100% finance. The model has seen 7 finance documents thousands of times each. It pattern-matches "total net sales" → Income Statement because it saw that exact document in training.
- Apple 2024 10K is completely unseen. The model has no memorized path. It's actually being asked to generalize, and 30% is what real generalization looks like right now.
- FinanceBench 76% is inflated. Most of those 114 correct answers are from documents the model has seen during DAgger training. It's not demonstrating skill — it's demonstrating recall.

Layer 2: The Navigator is Blind (No Content Signals)

The navigator receives: section title + page range. That's it. Zero content.

[0013] Consolidated Statements Of Operations (pages 32-33)
[0015] Consolidated Balance Sheets (pages 34-35)

To distinguish "total assets → Balance Sheet, not Income Statement", the model must know from training that balance sheets contain assets. For 39 memorized documents, this works. For Apple's unseen tree where some nodes are named Pages 28-31 (a StructDirect page-chunk fallback with zero semantic meaning), the model is navigating completely blind.

TreeGen had a critical advantage here that we lost: The LLM reading the document would write semantically rich titles. It might write "Balance Sheet: Total assets $364,980M, Liabilities $308,030M" as a node title. The navigator then has actual content signals. StructDirect gives you whatever the PDF's printed TOC says — which is good for SEC filings with standard names, but useless when a sub-section gets named Pages 28-31 as a fallback.

Layer 3: Only 18.9% of Training is Sub-Statement Level

From the data: 81.1% of training examples are top-level navigation decisions (which Item to go to). Only 18.9% involve navigating between sub-statements (Balance Sheet vs Income Statement vs Cash Flow).

But the Apple 10K failures are almost entirely sub-statement problems:
- Q01: Item 8 → picked wrong sub-section (index vs Income Statement)
- Q04: went to Income Statement for "total assets" (should be Balance Sheet)
- Q07: went to Income Statement for "Americas segment" (should be Item 7 MD&A)

The model has barely trained on this level of navigation. It knows "net income → go to Item 8". It has no concept of "total assets → go to Balance Sheet within Item 8".

---
TreeGen vs StructDirect — The Real Comparison

|                        | TreeGen                             | StructDirect             |
|------------------------|-------------------------------------|--------------------------|
| Speed                  | 3-15 min                            | <1 second                |
| Cost                   | $2-10/doc                           | $0                       |
| Section titles         | LLM-generated, content-descriptive  | Copied from PDF TOC      |
| Fallback for bad TOC   | Generates meaningful structure      | Pages 28-31 chunks       |
| Navigator signal       | High (titles describe content)      | Low (titles are labels)  |
| Unseen doc performance | Better generalization               | Depends on PDF quality   |

StructDirect is still the right choice — but it has a critical gap: when it falls back to page-chunking or gets generic TOC titles, it gives the navigator zero content signal. TreeGen's LLM could always generate a descriptive title even for generic sections.

---
The Real Fix — What "Think Bigger" Means Here

The professor is right. The mistake is optimizing navigation accuracy as the goal. Navigation is a means. The goal is: user uploads PDF, asks question, gets correct answer.

The system currently: Structure → Navigate (page range) → Read page text → Answer

The navigation step is broken for unseen documents because of blind navigation. The fix isn't training more — it's giving the navigator what it needs:

Each node needs a content preview. When StructDirect builds the tree, PyMuPDF can extract the first 150-200 characters of actual text from each section's starting page in milliseconds. Include that in the navigation prompt:

[0013] Consolidated Statements Of Operations (pages 32-33)
       "Apple Inc. CONSOLIDATED STATEMENTS OF OPERATIONS Net sales: Products $294,866  Services $96,169..."

[0015] Consolidated Balance Sheets (pages 34-35)
       "Apple Inc. CONSOLIDATED BALANCE SHEETS ASSETS Current assets: Cash and cash equivalents $29,943..."

Now the navigator can correctly route "total assets" → [0015] even for a completely unseen document, with zero retraining. The distinction becomes obvious from content, not from memorized title associations.

This is also what makes the system actually robust — it stops relying on title keyword matching and starts making content-informed decisions.

---
What Should We Actually Do?

The single highest-leverage change: Add content previews to StructDirect nodes at extraction time. Extract first 200 chars of page text for each node from PyMuPDF (it's already open, zero extra cost), pass them into the navigation prompt.

This fixes:
- Unseen document navigation (Apple 10K type failures)
- Generic node names (Pages 28-31 becomes meaningless — the preview shows what's actually there)
- Sub-statement confusion (model can see balance sheet keywords vs income statement keywords)

Before we rebuild training data or fine-tune v18, do you want to first test if this structural change alone (node previews in the navigation prompt) gets Apple 10K from 3/10 to something reasonable — using the existing v17 model, no retraining? That would tell us if the problem is the navigator's training or the input it's getting.
