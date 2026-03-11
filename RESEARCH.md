# PageIndex Deep Research

> Source: https://github.com/VectifyAI/PageIndex (branch: main)
> Fetched: 2026-03-10

---

## 1. Repository Structure

```
.claude/commands/dedupe.md
.github/workflows/autoclose-labeled-issues.yml
.github/workflows/backfill-dedupe.yml
.github/workflows/issue-dedupe.yml
.github/workflows/remove-autoclose-label.yml
.gitignore
CHANGELOG.md
LICENSE
README.md
cookbook/README.md
cookbook/agentic_retrieval.ipynb
cookbook/pageIndex_chat_quickstart.ipynb
cookbook/pageindex_RAG_simple.ipynb
cookbook/vision_RAG_pageindex.ipynb
pageindex/__init__.py
pageindex/config.yaml
pageindex/page_index.py
pageindex/page_index_md.py
pageindex/utils.py
requirements.txt
run_pageindex.py
scripts/autoclose-labeled-issues.js
scripts/comment-on-duplicates.sh
tests/pdfs/2023-annual-report-truncated.pdf
tests/pdfs/2023-annual-report.pdf
tests/pdfs/PRML.pdf
tests/pdfs/Regulation Best Interest_Interpretive release.pdf
tests/pdfs/Regulation Best Interest_proposed rule.pdf
tests/pdfs/earthmover.pdf
tests/pdfs/four-lectures.pdf
tests/pdfs/q1-fy25-earnings.pdf
tests/results/2023-annual-report-truncated_structure.json
tests/results/2023-annual-report_structure.json
tests/results/PRML_structure.json
tests/results/Regulation Best Interest_Interpretive release_structure.json
tests/results/Regulation Best Interest_proposed rule_structure.json
tests/results/earthmover_structure.json
tests/results/four-lectures_structure.json
tests/results/q1-fy25-earnings_structure.json
tutorials/doc-search/README.md
tutorials/doc-search/description.md
tutorials/doc-search/metadata.md
tutorials/doc-search/semantics.md
tutorials/tree-search/README.md
```

---

## 2. Tree Node Data Structure

### 2.1 JSON Schema (inferred from test outputs and code)

Every node in the output tree can contain the following fields:

```json
{
  "title": "<section title string>",
  "node_id": "<zero-padded 4-digit string, e.g. \"0000\">",
  "start_index": "<integer, 1-based physical page number where section starts>",
  "end_index": "<integer, 1-based physical page number where section ends (inclusive)>",
  "summary": "<LLM-generated summary of the section text (optional, if if_add_node_summary=yes)>",
  "prefix_summary": "<for markdown: summary of prefix text before child nodes (optional)>",
  "text": "<raw extracted text of the section (optional, if if_add_node_text=yes)>",
  "line_num": "<integer, line number in markdown (markdown mode only)>",
  "nodes": [ ...recursive child nodes... ]
}
```

**For intermediate TOC items (before post_processing):**

```json
{
  "structure": "<dot-notation hierarchy index, e.g. \"1.2.3\">",
  "title": "<section title>",
  "physical_index": "<integer page number>",
  "appear_start": "<\"yes\" or \"no\", whether section starts at beginning of its page>"
}
```

**Top-level output wrapper:**

```json
{
  "doc_name": "<filename without extension>",
  "doc_description": "<optional one-sentence LLM description of the entire document>",
  "structure": [ ...array of root TreeNode objects... ]
}
```

### 2.2 Concrete Example (earthmover.pdf)

```json
{
  "doc_name": "earthmover.pdf",
  "structure": [
    {
      "title": "Earth Mover's Distance based Similarity Search at Scale",
      "start_index": 1,
      "end_index": 1,
      "node_id": "0000"
    },
    {
      "title": "ABSTRACT",
      "start_index": 1,
      "end_index": 1,
      "node_id": "0001"
    },
    {
      "title": "INTRODUCTION",
      "start_index": 1,
      "end_index": 2,
      "node_id": "0002"
    },
    {
      "title": "PRELIMINARIES",
      "start_index": 2,
      "end_index": 2,
      "nodes": [
        {
          "title": "Computing the EMD",
          "start_index": 3,
          "end_index": 3,
          "node_id": "0004"
        },
        {
          "title": "Filter-and-Refinement Framework",
          "start_index": 3,
          "end_index": 4,
          "node_id": "0005"
        }
      ],
      "node_id": "0003"
    },
    {
      "title": "SCALING UP SSP",
      "start_index": 4,
      "end_index": 5,
      "node_id": "0006"
    },
    {
      "title": "BOOSTING THE REFINEMENT PHASE",
      "start_index": 5,
      "end_index": 5,
      "nodes": [
        { "title": "Analysis of EMD Calculation", "start_index": 5, "end_index": 6, "node_id": "0008" },
        { "title": "Progressive Bounding", "start_index": 6, "end_index": 6, "node_id": "0009" },
        { "title": "Sensitivity to Refinement Order", "start_index": 6, "end_index": 7, "node_id": "0010" },
        { "title": "Dynamic Refinement Ordering", "start_index": 7, "end_index": 8, "node_id": "0011" },
        { "title": "Running Upper Bound", "start_index": 8, "end_index": 8, "node_id": "0012" }
      ],
      "node_id": "0007"
    }
  ]
}
```

### 2.3 node_id Assignment (verbatim code from utils.py)

```python
def write_node_id(data, node_id=0):
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if 'nodes' in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for index in range(len(data)):
            node_id = write_node_id(data[index], node_id)
    return node_id
```

node_ids are assigned via depth-first pre-order traversal, zero-padded to 4 digits.

---

## 3. Tree Generation Algorithm

### 3.1 Entry Points

**`run_pageindex.py`** (CLI entry point):

```python
import argparse
import os
import json
from pageindex import *
from pageindex.page_index_md import md_to_tree

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-11-20', help='Model to use')
    parser.add_argument('--toc-check-pages', type=int, default=20)
    parser.add_argument('--max-pages-per-node', type=int, default=10)
    parser.add_argument('--max-tokens-per-node', type=int, default=20000)
    parser.add_argument('--if-add-node-id', type=str, default='yes')
    parser.add_argument('--if-add-node-summary', type=str, default='yes')
    parser.add_argument('--if-add-doc-description', type=str, default='no')
    parser.add_argument('--if-add-node-text', type=str, default='no')
    # Markdown specific
    parser.add_argument('--if-thinning', type=str, default='no')
    parser.add_argument('--thinning-threshold', type=int, default=5000)
    parser.add_argument('--summary-token-threshold', type=int, default=200)
    args = parser.parse_args()
```

**`pageindex/config.yaml`** (default configuration):

```yaml
model: "gpt-4o-2024-11-20"
toc_check_page_num: 20
max_page_num_each_node: 10
max_token_num_each_node: 20000
if_add_node_id: "yes"
if_add_node_summary: "yes"
if_add_doc_description: "no"
if_add_node_text: "no"
```

### 3.2 PDF Pipeline: `page_index_main()` (page_index.py)

```python
def page_index_main(doc, opt=None):
    logger = JsonLogger(doc)

    is_valid_pdf = (
        (isinstance(doc, str) and os.path.isfile(doc) and doc.lower().endswith(".pdf")) or
        isinstance(doc, BytesIO)
    )
    if not is_valid_pdf:
        raise ValueError("Unsupported input type. Expected a PDF file path or BytesIO object.")

    print('Parsing PDF...')
    page_list = get_page_tokens(doc)

    async def page_index_builder():
        structure = await tree_parser(page_list, opt, doc=doc, logger=logger)
        if opt.if_add_node_id == 'yes':
            write_node_id(structure)
        if opt.if_add_node_text == 'yes':
            add_node_text(structure, page_list)
        if opt.if_add_node_summary == 'yes':
            if opt.if_add_node_text == 'no':
                add_node_text(structure, page_list)
            await generate_summaries_for_structure(structure, model=opt.model)
            if opt.if_add_node_text == 'no':
                remove_structure_text(structure)
            if opt.if_add_doc_description == 'yes':
                clean_structure = create_clean_structure_for_description(structure)
                doc_description = generate_doc_description(clean_structure, model=opt.model)
                return {
                    'doc_name': get_pdf_name(doc),
                    'doc_description': doc_description,
                    'structure': structure,
                }
        return {
            'doc_name': get_pdf_name(doc),
            'structure': structure,
        }

    return asyncio.run(page_index_builder())
```

### 3.3 PDF Parsing (utils.py — `get_page_tokens`)

```python
def get_page_tokens(pdf_path, model="gpt-4o-2024-11-20", pdf_parser="PyPDF2"):
    enc = tiktoken.encoding_for_model(model)
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
    elif pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            pdf_stream = pdf_path
            doc = pymupdf.open(stream=pdf_stream, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
            doc = pymupdf.open(pdf_path)
        page_list = []
        for page in doc:
            page_text = page.get_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
```

Returns: list of `(page_text: str, token_count: int)` tuples, one per page.

### 3.4 `tree_parser()` — Main Orchestrator (page_index.py)

```python
async def tree_parser(page_list, opt, doc=None, logger=None):
    check_toc_result = check_toc(page_list, opt)
    logger.info(check_toc_result)

    if check_toc_result.get("toc_content") and check_toc_result["toc_content"].strip() and check_toc_result["page_index_given_in_toc"] == "yes":
        toc_with_page_number = await meta_processor(
            page_list,
            mode='process_toc_with_page_numbers',
            start_index=1,
            toc_content=check_toc_result['toc_content'],
            toc_page_list=check_toc_result['toc_page_list'],
            opt=opt,
            logger=logger)
    else:
        toc_with_page_number = await meta_processor(
            page_list,
            mode='process_no_toc',
            start_index=1,
            opt=opt,
            logger=logger)

    toc_with_page_number = add_preface_if_needed(toc_with_page_number)
    toc_with_page_number = await check_title_appearance_in_start_concurrent(toc_with_page_number, page_list, model=opt.model, logger=logger)

    valid_toc_items = [item for item in toc_with_page_number if item.get('physical_index') is not None]

    toc_tree = post_processing(valid_toc_items, len(page_list))
    tasks = [
        process_large_node_recursively(node, page_list, opt, logger=logger)
        for node in toc_tree
    ]
    await asyncio.gather(*tasks)

    return toc_tree
```

**Decision logic:**
1. Check if document has a TOC with page numbers → `process_toc_with_page_numbers`
2. If no TOC or TOC has no page numbers → `process_no_toc`
3. After getting flat TOC list, verify accuracy; fix errors; convert to tree

### 3.5 Three Processing Modes (`meta_processor`)

```python
async def meta_processor(page_list, mode=None, toc_content=None, toc_page_list=None, start_index=1, opt=None, logger=None):
    if mode == 'process_toc_with_page_numbers':
        toc_with_page_number = process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=opt.toc_check_page_num, model=opt.model, logger=logger)
    elif mode == 'process_toc_no_page_numbers':
        toc_with_page_number = process_toc_no_page_numbers(toc_content, toc_page_list, page_list, model=opt.model, logger=logger)
    else:
        toc_with_page_number = process_no_toc(page_list, start_index=start_index, model=opt.model, logger=logger)

    toc_with_page_number = [item for item in toc_with_page_number if item.get('physical_index') is not None]

    toc_with_page_number = validate_and_truncate_physical_indices(
        toc_with_page_number,
        len(page_list),
        start_index=start_index,
        logger=logger
    )

    accuracy, incorrect_results = await verify_toc(page_list, toc_with_page_number, start_index=start_index, model=opt.model)

    if accuracy == 1.0 and len(incorrect_results) == 0:
        return toc_with_page_number
    if accuracy > 0.6 and len(incorrect_results) > 0:
        toc_with_page_number, incorrect_results = await fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results, start_index=start_index, max_attempts=3, model=opt.model, logger=logger)
        return toc_with_page_number
    else:
        # Fallback cascade:
        if mode == 'process_toc_with_page_numbers':
            return await meta_processor(page_list, mode='process_toc_no_page_numbers', ...)
        elif mode == 'process_toc_no_page_numbers':
            return await meta_processor(page_list, mode='process_no_toc', ...)
        else:
            raise Exception('Processing failed')
```

**Fallback cascade:** `process_toc_with_page_numbers` → `process_toc_no_page_numbers` → `process_no_toc`

### 3.6 All LLM Prompts Used in Tree Generation

#### Prompt: TOC Detection

```python
def toc_detector_single_page(content, model=None):
    prompt = f"""
    Your job is to detect if there is a table of content provided in the given text.

    Given text: {content}

    return the following JSON format:
    {{
        "thinking": <why do you think there is a table of content in the given text>
        "toc_detected": "<yes or no>",
    }}

    Directly return the final JSON structure. Do not output anything else.
    Please note: abstract,summary, notation list, figure list, table list, etc. are not table of contents."""
```

#### Prompt: Check if TOC Extraction is Complete

```python
def check_if_toc_extraction_is_complete(content, toc, model=None):
    prompt = f"""
    You are given a partial document  and a  table of contents.
    Your job is to check if the  table of contents is complete, which it contains all the main sections in the partial document.

    Reply format:
    {{
        "thinking": <why do you think the table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else."""
    prompt = prompt + '\n Document:\n' + content + '\n Table of contents:\n' + toc
```

#### Prompt: Check if TOC Transformation is Complete

```python
def check_if_toc_transformation_is_complete(content, toc, model=None):
    prompt = f"""
    You are given a raw table of contents and a  table of contents.
    Your job is to check if the  table of contents is complete.

    Reply format:
    {{
        "thinking": <why do you think the cleaned table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else."""
    prompt = prompt + '\n Raw Table of contents:\n' + content + '\n Cleaned Table of contents:\n' + toc
```

#### Prompt: Extract TOC Content

```python
def extract_toc_content(content, model=None):
    prompt = f"""
    Your job is to extract the full table of contents from the given text, replace ... with :

    Given text: {content}

    Directly return the full table of contents content. Do not output anything else."""
```

#### Prompt: Detect Page Index in TOC

```python
def detect_page_index(toc_content, model=None):
    prompt = f"""
    You will be given a table of contents.

    Your job is to detect if there are page numbers/indices given within the table of contents.

    Given text: {toc_content}

    Reply format:
    {{
        "thinking": <why do you think there are page numbers/indices given within the table of contents>
        "page_index_given_in_toc": "<yes or no>"
    }}
    Directly return the final JSON structure. Do not output anything else."""
```

#### Prompt: TOC Transformer (TOC text → JSON)

```python
def toc_transformer(toc_content, model=None):
    init_prompt = """
    You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

    structure is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format:
    {
    table_of_contents: [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "page": <page number or None>,
        },
        ...
        ],
    }
    You should transform the full table of contents in one go.
    Directly return the final JSON structure, do not output anything else. """
    prompt = init_prompt + '\n Given table of contents\n:' + toc_content
```

Includes continuation logic for truncated outputs via chat history.

#### Prompt: TOC Index Extractor (map TOC items → physical page numbers)

```python
def toc_index_extractor(toc, content, model=None):
    toc_extractor_prompt = """
    You are given a table of contents in a json format and several pages of a document, your job is to add the physical_index to the table of contents in the json format.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format:
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "physical_index": "<physical_index_X>" (keep the format)
        },
        ...
    ]

    Only add the physical_index to the sections that are in the provided pages.
    If the section is not in the provided pages, do not add the physical_index to it.
    Directly return the final JSON structure. Do not output anything else."""
    prompt = toc_extractor_prompt + '\nTable of contents:\n' + str(toc) + '\nDocument pages:\n' + content
```

#### Prompt: Add Page Number to TOC (no-page-number mode)

```python
def add_page_number_to_toc(part, structure, model=None):
    fill_prompt_seq = """
    You are given an JSON structure of a document and a partial part of the document. Your task is to check if the title that is described in the structure is started in the partial given document.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    If the full target section starts in the partial given document, insert the given JSON structure with the "start": "yes", and "start_index": "<physical_index_X>".

    If the full target section does not start in the partial given document, insert "start": "no",  "start_index": None.

    The response should be in the following format.
        [
            {
                "structure": <structure index, "x.x.x" or None> (string),
                "title": <title of the section>,
                "start": "<yes or no>",
                "physical_index": "<physical_index_X> (keep the format)" or None
            },
            ...
        ]
    The given structure contains the result of the previous part, you need to fill the result of the current part, do not change the previous result.
    Directly return the final JSON structure. Do not output anything else."""
    prompt = fill_prompt_seq + f"\n\nCurrent Partial Document:\n{part}\n\nGiven Structure\n{json.dumps(structure, indent=2)}\n"
```

#### Prompt: Generate TOC from Scratch (no-TOC mode) — Initial Part

```python
def generate_toc_init(part, model=None):
    prompt = """
    You are an expert in extracting hierarchical tree structure, your task is to generate the tree structure of the document.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X.

    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format.
        [
            {{
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            }},

        ],


    Directly return the final JSON structure. Do not output anything else."""
    prompt = prompt + '\nGiven text\n:' + part
```

#### Prompt: Generate TOC from Scratch — Continuation

```python
def generate_toc_continue(toc_content, part, model="gpt-4o-2024-11-20"):
    prompt = """
    You are an expert in extracting hierarchical tree structure.
    You are given a tree structure of the previous part and the text of the current part.
    Your task is to continue the tree structure from the previous part to include the current part.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X. \

    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format.
        [
            {
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            },
            ...
        ]

    Directly return the additional part of the final JSON structure. Do not output anything else."""
    prompt = prompt + '\nGiven text\n:' + part + '\nPrevious tree structure\n:' + json.dumps(toc_content, indent=2)
```

#### Prompt: Check Title Appearance in Page (verification)

```python
async def check_title_appearance(item, page_list, start_index=1, model=None):
    title = item['title']
    page_number = item['physical_index']
    page_text = page_list[page_number-start_index][0]

    prompt = f"""
    Your job is to check if the given section appears or starts in the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.

    Reply format:
    {{

        "thinking": <why do you think the section appears or starts in the page_text>
        "answer": "yes or no" (yes if the section appears or starts in the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not output anything else."""
```

#### Prompt: Check Title Appearance at Start of Page

```python
async def check_title_appearance_in_start(title, page_text, model=None, logger=None):
    prompt = f"""
    You will be given the current section title and the current page_text.
    Your job is to check if the current section starts in the beginning of the given page_text.
    If there are other contents before the current section title, then the current section does not start in the beginning of the given page_text.
    If the current section title is the first content in the given page_text, then the current section starts in the beginning of the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.

    reply format:
    {{
        "thinking": <why do you think the section appears or starts in the page_text>
        "start_begin": "yes or no" (yes if the section starts in the beginning of the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not output anything else."""
```

#### Prompt: Fix Incorrect TOC Entry

```python
def single_toc_item_index_fixer(section_title, content, model="gpt-4o-2024-11-20"):
    toc_extractor_prompt = """
    You are given a section title and several pages of a document, your job is to find the physical index of the start page of the section in the partial document.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    Reply in a JSON format:
    {
        "thinking": <explain which page, started and closed by <physical_index_X>, contains the start of this section>,
        "physical_index": "<physical_index_X>" (keep the format)
    }
    Directly return the final JSON structure. Do not output anything else."""
    prompt = toc_extractor_prompt + '\nSection Title:\n' + str(section_title) + '\nDocument pages:\n' + content
```

#### Prompt: Generate Node Summary

```python
async def generate_node_summary(node, model=None):
    prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

    Partial Document Text: {node['text']}

    Directly return the description, do not include any other text.
    """
```

#### Prompt: Generate Document Description

```python
def generate_doc_description(structure, model=None):
    prompt = f"""Your are an expert in generating descriptions for a document.
    You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.

    Document Structure: {structure}

    Directly return the description, do not include any other text.
    """
```

### 3.7 `post_processing()` — Flat List to Tree

```python
def post_processing(structure, end_physical_index):
    # Convert page_number to start_index in flat list
    for i, item in enumerate(structure):
        item['start_index'] = item.get('physical_index')
        if i < len(structure) - 1:
            if structure[i + 1].get('appear_start') == 'yes':
                item['end_index'] = structure[i + 1]['physical_index']-1
            else:
                item['end_index'] = structure[i + 1]['physical_index']
        else:
            item['end_index'] = end_physical_index
    tree = list_to_tree(structure)
    if len(tree) != 0:
        return tree
    else:
        for node in structure:
            node.pop('appear_start', None)
            node.pop('physical_index', None)
        return structure
```

`end_index` of a section = `start_index` of the next section minus 1 if the next section starts at the very beginning of its page (`appear_start == 'yes'`), otherwise equals the next section's `start_index`.

### 3.8 `list_to_tree()` — Hierarchy Building from Dot-Notation

```python
def list_to_tree(data):
    def get_parent_structure(structure):
        if not structure:
            return None
        parts = str(structure).split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else None

    nodes = {}
    root_nodes = []

    for item in data:
        structure = item.get('structure')
        node = {
            'title': item.get('title'),
            'start_index': item.get('start_index'),
            'end_index': item.get('end_index'),
            'nodes': []
        }
        nodes[structure] = node
        parent_structure = get_parent_structure(structure)

        if parent_structure:
            if parent_structure in nodes:
                nodes[parent_structure]['nodes'].append(node)
            else:
                root_nodes.append(node)
        else:
            root_nodes.append(node)

    def clean_node(node):
        if not node['nodes']:
            del node['nodes']
        else:
            for child in node['nodes']:
                clean_node(child)
        return node

    return [clean_node(node) for node in root_nodes]
```

---

## 4. Tree Search / Retrieval Algorithm

### 4.1 Basic Tree Search Prompt (from tutorials/tree-search/README.md)

```python
prompt = f"""
You are given a query and the tree structure of a document.
You need to find all nodes that are likely to contain the answer.

Query: {query}

Document tree structure: {PageIndex_Tree}

Reply in the following JSON format:
{{
  "thinking": <your reasoning about which nodes are relevant>,
  "node_list": [node_id1, node_id2, ...]
}}
"""
```

### 4.2 Tree Search with Expert Knowledge / Preference

```python
prompt = f"""
You are given a question and a tree structure of a document.
You need to find all nodes that are likely to contain the answer.

Query: {query}

Document tree structure:  {PageIndex_Tree}

Expert Knowledge of relevant sections: {Preference}

Reply in the following JSON format:
{{
  "thinking": <reasoning about which nodes are relevant>,
  "node_list": [node_id1, node_id2, ...]
}}
"""
```

**Example expert preference:**
> If the query mentions EBITDA adjustments, prioritize Item 7 (MD&A) and footnotes in Item 8 (Financial Statements) in 10-K reports.

### 4.3 Cookbook Tree Search Prompt (pageindex_RAG_simple.ipynb — verbatim)

```python
query = "What are the conclusions in this document?"

tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])

search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{json.dumps(tree_without_text, indent=2)}

Please reply in the following JSON format:
{{
    "thinking": "<Your thinking process on which nodes are relevant to the question>",
    "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
}}
Directly return the final JSON structure. Do not output anything else.
"""

tree_search_result = await call_llm(search_prompt)
```

### 4.4 Answer Generation Prompt (verbatim from cookbook)

```python
answer_prompt = f"""
Answer the question based on the context:

Question: {query}
Context: {relevant_content}

Provide a clear, concise answer based only on the context provided.
"""
answer = await call_llm(answer_prompt)
```

### 4.5 Agentic Retrieval Prompt (agentic_retrieval.ipynb — verbatim)

```python
retrieval_prompt = f"""
Your job is to retrieve the raw relevant content from the document based on the user's query.

Query: {query}

Return in JSON format:
```json
[
  {{
    "page": <number>,
    "content": "<raw text>"
  }},
  ...
]
```
"""
```

This prompt is sent to the PageIndex Chat API (cloud), which internally handles tree search and returns structured retrieved results.

### 4.6 Node Retrieval Pattern (full code from cookbook)

```python
# Step 1: get tree
tree = pi_client.get_tree(doc_id, node_summary=True)['result']

# Step 2: run tree search
tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])
tree_search_result = await call_llm(search_prompt)  # returns JSON with node_list

# Step 3: extract text from retrieved nodes
node_map = utils.create_node_mapping(tree)
node_list = json.loads(tree_search_result)["node_list"]
relevant_content = "\n\n".join(node_map[node_id]["text"] for node_id in node_list)

# Step 4: generate answer
answer = await call_llm(answer_prompt)
```

### 4.7 Cloud API Notes

From the tree-search tutorial:
> In our dashboard and retrieval API, we use a combination of LLM tree search and value function-based Monte Carlo Tree Search (MCTS). More details will be released soon.

The open-source version uses only direct LLM prompting for tree search.

---

## 5. Long Document Handling

### 5.1 Page Grouping for Context Window Management

```python
def page_list_to_group_text(page_contents, token_lengths, max_tokens=20000, overlap_page=1):
    num_tokens = sum(token_lengths)

    if num_tokens <= max_tokens:
        # merge all pages into one text
        page_text = "".join(page_contents)
        return [page_text]

    subsets = []
    current_subset = []
    current_token_count = 0

    expected_parts_num = math.ceil(num_tokens / max_tokens)
    average_tokens_per_part = math.ceil(((num_tokens / expected_parts_num) + max_tokens) / 2)

    for i, (page_content, page_tokens) in enumerate(zip(page_contents, token_lengths)):
        if current_token_count + page_tokens > average_tokens_per_part:
            subsets.append(''.join(current_subset))
            # Start new subset from overlap if specified
            overlap_start = max(i - overlap_page, 0)
            current_subset = page_contents[overlap_start:i]
            current_token_count = sum(token_lengths[overlap_start:i])

        current_subset.append(page_content)
        current_token_count += page_tokens

    if current_subset:
        subsets.append(''.join(current_subset))

    print('divide page_list to groups', len(subsets))
    return subsets
```

**Key design:** `average_tokens_per_part = ceil((num_tokens/N + max_tokens) / 2)` — the target per-chunk size is the average of the naive split size and the max, with 1 page of overlap between chunks.

### 5.2 No-TOC Mode: Sequential Generation Across Chunks

```python
def process_no_toc(page_list, start_index=1, model=None, logger=None):
    page_contents = []
    token_lengths = []
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    group_texts = page_list_to_group_text(page_contents, token_lengths)

    # First chunk: initialize TOC
    toc_with_page_number = generate_toc_init(group_texts[0], model)
    # Subsequent chunks: continue from previous result
    for group_text in group_texts[1:]:
        toc_with_page_number_additional = generate_toc_continue(toc_with_page_number, group_text, model)
        toc_with_page_number.extend(toc_with_page_number_additional)

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    return toc_with_page_number
```

### 5.3 No-Page-Number TOC Mode: Sequential Page Mapping

```python
def process_toc_no_page_numbers(toc_content, toc_page_list, page_list, start_index=1, model=None, logger=None):
    toc_content = toc_transformer(toc_content, model)
    page_contents = []
    token_lengths = []
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))

    group_texts = page_list_to_group_text(page_contents, token_lengths)

    toc_with_page_number = copy.deepcopy(toc_content)
    for group_text in group_texts:
        toc_with_page_number = add_page_number_to_toc(group_text, toc_with_page_number, model)

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    return toc_with_page_number
```

### 5.4 Recursive Large-Node Subdivision

Nodes that exceed `max_page_num_each_node=10` pages AND `max_token_num_each_node=20000` tokens get recursively re-processed:

```python
async def process_large_node_recursively(node, page_list, opt=None, logger=None):
    node_page_list = page_list[node['start_index']-1:node['end_index']]
    token_num = sum([page[1] for page in node_page_list])

    if node['end_index'] - node['start_index'] > opt.max_page_num_each_node and token_num >= opt.max_token_num_each_node:
        print('large node:', node['title'], 'start_index:', node['start_index'], 'end_index:', node['end_index'], 'token_num:', token_num)

        node_toc_tree = await meta_processor(node_page_list, mode='process_no_toc', start_index=node['start_index'], opt=opt, logger=logger)
        node_toc_tree = await check_title_appearance_in_start_concurrent(node_toc_tree, page_list, model=opt.model, logger=logger)

        valid_node_toc_items = [item for item in node_toc_tree if item.get('physical_index') is not None]

        if valid_node_toc_items and node['title'].strip() == valid_node_toc_items[0]['title'].strip():
            node['nodes'] = post_processing(valid_node_toc_items[1:], node['end_index'])
            node['end_index'] = valid_node_toc_items[1]['start_index'] if len(valid_node_toc_items) > 1 else node['end_index']
        else:
            node['nodes'] = post_processing(valid_node_toc_items, node['end_index'])
            node['end_index'] = valid_node_toc_items[0]['start_index'] if valid_node_toc_items else node['end_index']

    if 'nodes' in node and node['nodes']:
        tasks = [
            process_large_node_recursively(child_node, page_list, opt, logger=logger)
            for child_node in node['nodes']
        ]
        await asyncio.gather(*tasks)

    return node
```

### 5.5 Page Tagging Format

Pages are injected into prompts with XML-like tags:

```
<physical_index_5>
[page text here]
<physical_index_5>
```

Both the opening and closing tag use the same tag (not open/close pairing). The LLM is instructed to reference these tags when identifying section start pages.

### 5.6 TOC Continuation for Truncated Output

When an LLM response is truncated (finish_reason = "length"), the code continues generation via chat history:

```python
def extract_toc_content(content, model=None):
    response, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt)

    if_complete = check_if_toc_transformation_is_complete(content, response, model)
    if if_complete == "yes" and finish_reason == "finished":
        return response

    chat_history = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    prompt = f"""please continue the generation of table of contents , directly output the remaining part of the structure"""
    new_response, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt, chat_history=chat_history)
    response = response + new_response
    # ... loops until complete or max retries (5 attempts)
    if len(chat_history) > 5:
        raise Exception('Failed to complete table of contents after maximum retries')
```

---

## 6. OCR Pipeline

PageIndex does NOT use traditional OCR. It uses:

1. **PyPDF2** (default) or **PyMuPDF** for text extraction from PDFs — text layer only
2. **Vision-RAG approach** for scanned/image PDFs: uses a VLM (GPT-4.1 with vision) to process page images directly

### 6.1 Text Extraction (utils.py)

```python
def get_page_tokens(pdf_path, model="gpt-4o-2024-11-20", pdf_parser="PyPDF2"):
    enc = tiktoken.encoding_for_model(model)
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
    elif pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            doc = pymupdf.open(stream=pdf_stream, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
            doc = pymupdf.open(pdf_path)
        page_list = []
        for page in doc:
            page_text = page.get_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
```

### 6.2 Vision-Based Pipeline (vision_RAG_pageindex.ipynb — verbatim)

For image-heavy documents, the cookbook implements a VLM pipeline that bypasses OCR entirely:

```python
def extract_pdf_page_images(pdf_path, output_dir="pdf_images"):
    os.makedirs(output_dir, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    page_images = {}
    total_pages = len(pdf_document)
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        # Convert page to image
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("jpeg")
        image_path = os.path.join(output_dir, f"page_{page_number + 1}.jpg")
        with open(image_path, "wb") as image_file:
            image_file.write(img_data)
        page_images[page_number + 1] = image_path
    pdf_document.close()
    return page_images, total_pages


def get_page_images_for_nodes(node_list, node_map, page_images):
    image_paths = []
    seen_pages = set()
    for node_id in node_list:
        node_info = node_map[node_id]
        for page_num in range(node_info['start_index'], node_info['end_index'] + 1):
            if page_num not in seen_pages:
                image_paths.append(page_images[page_num])
                seen_pages.add(page_num)
    return image_paths
```

**VLM Answer Generation (vision_RAG_pageindex.ipynb — verbatim):**

```python
async def call_vlm(prompt, image_paths=None, model="gpt-4.1"):
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    messages = [{"role": "user", "content": prompt}]
    if image_paths:
        content = [{"type": "text", "text": prompt}]
        for image in image_paths:
            if os.path.exists(image):
                with open(image, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    })
        messages[0]["content"] = content
    response = await client.chat.completions.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message.content.strip()
```

**Vision answer prompt (verbatim):**

```python
answer_prompt = f"""
Answer the question based on the images of the document pages as context.

Question: {query}

Provide a clear, concise answer based only on the context provided.
"""
answer = await call_vlm(answer_prompt, retrieved_page_images)
```

### 6.3 Vision-RAG Full Flow

1. Build PageIndex tree from PDF text (for structure/navigation only)
2. Run tree search with LLM using node summaries (text-based)
3. Extract PDF page images for the retrieved node page ranges
4. Send page images to VLM for answer generation (no OCR step)

---

## 7. Cookbook Examples

### 7.1 pageindex_RAG_simple.ipynb — Vectorless RAG (all cells verbatim)

**Cell 1 (Markdown):**
```
![pageindex_banner](https://pageindex.ai/static/images/pageindex_banner.jpg)
```

**Cell 2 (Markdown):**
```
# Simple Vectorless RAG with PageIndex
```

**Cell 3 (Markdown):**
```
## PageIndex Introduction
PageIndex is a new **reasoning-based**, **vectorless RAG** framework that performs retrieval in two steps:
1. Generate a tree structure index of documents
2. Perform reasoning-based retrieval through tree search

Compared to traditional vector-based RAG, PageIndex features:
- **No Vectors Needed**: Uses document structure and LLM reasoning for retrieval.
- **No Chunking Needed**: Documents are organized into natural sections rather than artificial chunks.
- **Human-like Retrieval**: Simulates how human experts navigate and extract knowledge from complex documents.
- **Transparent Retrieval Process**: Retrieval based on reasoning — say goodbye to approximate semantic search ("vibe retrieval").
```

**Cell 4 (Code) — Install:**
```python
%pip install -q --upgrade pageindex
```

**Cell 5 (Code) — Setup:**
```python
from pageindex import PageIndexClient
import pageindex.utils as utils

# Get your PageIndex API key from https://dash.pageindex.ai/api-keys
PAGEINDEX_API_KEY = "YOUR_PAGEINDEX_API_KEY"
pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
```

**Cell 6 (Code) — LLM setup:**
```python
import openai
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

async def call_llm(prompt, model="gpt-4.1", temperature=0):
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()
```

**Cell 7 (Code) — Submit document:**
```python
import os, requests

pdf_url = "https://arxiv.org/pdf/2501.12948.pdf"
pdf_path = os.path.join("../data", pdf_url.split('/')[-1])
os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

response = requests.get(pdf_url)
with open(pdf_path, "wb") as f:
    f.write(response.content)
print(f"Downloaded {pdf_url}")

doc_id = pi_client.submit_document(pdf_path)["doc_id"]
print('Document Submitted:', doc_id)
```

**Cell 8 (Code) — Get tree:**
```python
if pi_client.is_retrieval_ready(doc_id):
    tree = pi_client.get_tree(doc_id, node_summary=True)['result']
    print('Simplified Tree Structure of the Document:')
    utils.print_tree(tree)
else:
    print("Processing document, please try again later...")
```

**Cell 9 (Code) — Tree search:**
```python
import json

query = "What are the conclusions in this document?"

tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])

search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{json.dumps(tree_without_text, indent=2)}

Please reply in the following JSON format:
{{
    "thinking": "<Your thinking process on which nodes are relevant to the question>",
    "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
}}
Directly return the final JSON structure. Do not output anything else.
"""

tree_search_result = await call_llm(search_prompt)
```

**Cell 10 (Code) — Print retrieved nodes:**
```python
node_map = utils.create_node_mapping(tree)
tree_search_result_json = json.loads(tree_search_result)

print('Reasoning Process:')
utils.print_wrapped(tree_search_result_json['thinking'])

print('\nRetrieved Nodes:')
for node_id in tree_search_result_json["node_list"]:
    node = node_map[node_id]
    print(f"Node ID: {node['node_id']}\t Page: {node['page_index']}\t Title: {node['title']}")
```

**Cell 11 (Code) — Extract context:**
```python
node_list = json.loads(tree_search_result)["node_list"]
relevant_content = "\n\n".join(node_map[node_id]["text"] for node_id in node_list)

print('Retrieved Context:\n')
utils.print_wrapped(relevant_content[:1000] + '...')
```

**Cell 12 (Code) — Generate answer:**
```python
answer_prompt = f"""
Answer the question based on the context:

Question: {query}
Context: {relevant_content}

Provide a clear, concise answer based only on the context provided.
"""

print('Generated Answer:\n')
answer = await call_llm(answer_prompt)
utils.print_wrapped(answer)
```

---

### 7.2 vision_RAG_pageindex.ipynb — Vision RAG (all cells verbatim)

**Cell 1 (Markdown):**
```
# A Vision-based, Vectorless RAG System for Long Documents
```

**Cell 2 (Markdown):**
```
In modern document question answering (QA) systems, Optical Character Recognition (OCR) serves an important role
by converting PDF pages into text that can be processed by Large Language Models (LLMs)...

> **If a VLM can already process both the document images and the query to produce an answer directly, do we still
> need the intermediate OCR step?**

In this notebook, we give a practical implementation of a vision-based question-answering system for long documents,
without relying on OCR. Specifically, we use PageIndex as a reasoning-based retrieval layer and OpenAI's multimodal
GPT-4.1 as the VLM for visual reasoning and answer generation.
```

**Cell 3 (Code) — Install:**
```python
%pip install -q --upgrade pageindex requests openai PyMuPDF
```

**Cell 4 (Code) — Setup PageIndex:**
```python
from pageindex import PageIndexClient
import pageindex.utils as utils

PAGEINDEX_API_KEY = "YOUR_PAGEINDEX_API_KEY"
pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
```

**Cell 5 (Code) — Setup VLM:**
```python
import openai, fitz, base64, os

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

async def call_vlm(prompt, image_paths=None, model="gpt-4.1"):
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    messages = [{"role": "user", "content": prompt}]
    if image_paths:
        content = [{"type": "text", "text": prompt}]
        for image in image_paths:
            if os.path.exists(image):
                with open(image, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    })
        messages[0]["content"] = content
    response = await client.chat.completions.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message.content.strip()
```

**Cell 6 (Code) — PDF image extraction helpers:**
```python
def extract_pdf_page_images(pdf_path, output_dir="pdf_images"):
    os.makedirs(output_dir, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    page_images = {}
    total_pages = len(pdf_document)
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("jpeg")
        image_path = os.path.join(output_dir, f"page_{page_number + 1}.jpg")
        with open(image_path, "wb") as image_file:
            image_file.write(img_data)
        page_images[page_number + 1] = image_path
        print(f"Saved page {page_number + 1} image: {image_path}")
    pdf_document.close()
    return page_images, total_pages

def get_page_images_for_nodes(node_list, node_map, page_images):
    image_paths = []
    seen_pages = set()
    for node_id in node_list:
        node_info = node_map[node_id]
        for page_num in range(node_info['start_index'], node_info['end_index'] + 1):
            if page_num not in seen_pages:
                image_paths.append(page_images[page_num])
                seen_pages.add(page_num)
    return image_paths
```

**Cell 7 (Code) — Submit document:**
```python
import os, requests

pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"  # the "Attention Is All You Need" paper
pdf_path = os.path.join("../data", pdf_url.split('/')[-1])
os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

response = requests.get(pdf_url)
with open(pdf_path, "wb") as f:
    f.write(response.content)
print(f"Downloaded {pdf_url}\n")

print("Extracting page images...")
page_images, total_pages = extract_pdf_page_images(pdf_path)
print(f"Extracted {len(page_images)} page images from {total_pages} total pages.\n")

doc_id = pi_client.submit_document(pdf_path)["doc_id"]
print('Document Submitted:', doc_id)
```

**Cell 8 (Code) — Get tree:**
```python
if pi_client.is_retrieval_ready(doc_id):
    tree = pi_client.get_tree(doc_id, node_summary=True)['result']
    print('Simplified Tree Structure of the Document:')
    utils.print_tree(tree, exclude_fields=['text'])
else:
    print("Processing document, please try again later...")
```

**Cell 9 (Code) — Tree search:**
```python
import json

query = "What is the last operation in the Scaled Dot-Product Attention figure?"

tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])

search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all tree nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{json.dumps(tree_without_text, indent=2)}

Please reply in the following JSON format:
{{
    "thinking": "<Your thinking process on which nodes are relevant to the question>",
    "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
}}
Directly return the final JSON structure. Do not output anything else.
"""

tree_search_result = await call_vlm(search_prompt)
```

**Cell 10 (Code) — Print retrieved nodes:**
```python
node_map = utils.create_node_mapping(tree, include_page_ranges=True, max_page=total_pages)
tree_search_result_json = json.loads(tree_search_result)

print('Reasoning Process:\n')
utils.print_wrapped(tree_search_result_json['thinking'])

print('\nRetrieved Nodes:\n')
for node_id in tree_search_result_json["node_list"]:
    node_info = node_map[node_id]
    node = node_info['node']
    start_page = node_info['start_index']
    end_page = node_info['end_index']
    page_range = start_page if start_page == end_page else f"{start_page}-{end_page}"
    print(f"Node ID: {node['node_id']}\t Pages: {page_range}\t Title: {node['title']}")
```

**Cell 11 (Code) — Get page images for retrieved nodes:**
```python
retrieved_nodes = tree_search_result_json["node_list"]
retrieved_page_images = get_page_images_for_nodes(retrieved_nodes, node_map, page_images)
print(f'\nRetrieved {len(retrieved_page_images)} PDF page image(s) for visual context.')
```

**Cell 12 (Code) — Generate answer with VLM:**
```python
answer_prompt = f"""
Answer the question based on the images of the document pages as context.

Question: {query}

Provide a clear, concise answer based only on the context provided.
"""

print('Generated answer using VLM with retrieved PDF page images as visual context:\n')
answer = await call_vlm(answer_prompt, retrieved_page_images)
utils.print_wrapped(answer)
```

---

### 7.3 agentic_retrieval.ipynb — Agentic Retrieval (all cells verbatim)

**Cell 1 (Code) — Install:**
```python
%pip install -q --upgrade pageindex
```

**Cell 2 (Code) — Setup:**
```python
from pageindex import PageIndexClient

PAGEINDEX_API_KEY = "YOUR_PAGEINDEX_API_KEY"
pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
```

**Cell 3 (Code) — Upload document:**
```python
import os, requests

pdf_url = "https://arxiv.org/pdf/2507.13334.pdf"
pdf_path = os.path.join("../data", pdf_url.split('/')[-1])
os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

response = requests.get(pdf_url)
with open(pdf_path, "wb") as f:
    f.write(response.content)
print(f"Downloaded {pdf_url}")

doc_id = pi_client.submit_document(pdf_path)["doc_id"]
print('Document Submitted:', doc_id)
```

**Cell 4 (Code) — Check status:**
```python
from pprint import pprint

doc_info = pi_client.get_document(doc_id)
pprint(doc_info)

if doc_info['status'] == 'completed':
  print(f"\n Document ready! ({doc_info['pageNum']} pages)")
elif doc_info['status'] == 'processing':
  print("\n Document is still processing. Please wait and check again.")
```

**Cell 5 (Code) — Direct QA chat:**
```python
query = "What are the evaluation methods used in this paper?"

for chunk in pi_client.chat_completions(
    messages=[{"role": "user", "content": query}],
    doc_id=doc_id,
    stream=True
):
    print(chunk, end='', flush=True)
```

**Cell 6 (Code) — Agentic retrieval prompt:**
```python
retrieval_prompt = f"""
Your job is to retrieve the raw relevant content from the document based on the user's query.

Query: {query}

Return in JSON format:
```json
[
  {{
    "page": <number>,
    "content": "<raw text>"
  }},
  ...
]
```
"""

full_response = ""

for chunk in pi_client.chat_completions(
    messages=[{"role": "user", "content": retrieval_prompt}],
    doc_id=doc_id,
    stream=True
):
    print(chunk, end='', flush=True)
    full_response += chunk
```

**Cell 7 (Code) — Extract JSON results:**
```python
%pip install -q jsonextractor

def extract_json(content):
    from json_extractor import JsonExtractor
    start_idx = content.find("```json")
    if start_idx != -1:
        start_idx += 7
        end_idx = content.rfind("```")
        json_content = content[start_idx:end_idx].strip()
    return JsonExtractor.extract_valid_json(json_content)

from pprint import pprint
pprint(extract_json(full_response))
```

---

## 8. Documentation Analysis

### 8.1 Configuration Parameters (config.yaml — complete)

```yaml
model: "gpt-4o-2024-11-20"
toc_check_page_num: 20
max_page_num_each_node: 10
max_token_num_each_node: 20000
if_add_node_id: "yes"
if_add_node_summary: "yes"
if_add_doc_description: "no"
if_add_node_text: "no"
```

| Parameter | Default | Description |
|---|---|---|
| `model` | `gpt-4o-2024-11-20` | LLM used for all prompts |
| `toc_check_page_num` | `20` | First N pages to scan for TOC |
| `max_page_num_each_node` | `10` | Threshold: split nodes larger than this |
| `max_token_num_each_node` | `20000` | Threshold: split nodes with more tokens |
| `if_add_node_id` | `yes` | Add `node_id` field to each node |
| `if_add_node_summary` | `yes` | Add LLM-generated `summary` to each node |
| `if_add_doc_description` | `no` | Add one-sentence doc description |
| `if_add_node_text` | `no` | Include raw page text in output |

### 8.2 Python API

```python
# Simple API
from pageindex import page_index

result = page_index(
    doc="./document.pdf",           # PDF path or BytesIO
    model="gpt-4o-2024-11-20",
    toc_check_page_num=20,
    max_page_num_each_node=10,
    max_token_num_each_node=20000,
    if_add_node_id="yes",
    if_add_node_summary="yes",
    if_add_doc_description="no",
    if_add_node_text="no"
)
# result = {"doc_name": ..., "structure": [...]}

# Markdown API (async)
from pageindex.page_index_md import md_to_tree
import asyncio

result = asyncio.run(md_to_tree(
    md_path="./document.md",
    if_thinning=False,
    min_token_threshold=5000,
    if_add_node_summary='yes',
    summary_token_threshold=200,
    model="gpt-4o-2024-11-20",
    if_add_doc_description='no',
    if_add_node_text='no',
    if_add_node_id='yes'
))
```

### 8.3 Cloud SDK API (PageIndexClient)

```python
from pageindex import PageIndexClient
pi_client = PageIndexClient(api_key="YOUR_API_KEY")

# Submit document
result = pi_client.submit_document("./document.pdf")
doc_id = result["doc_id"]

# Check status
status = pi_client.get_document(doc_id)["status"]  # "processing" | "completed"

# Check if ready
is_ready = pi_client.is_retrieval_ready(doc_id)  # bool

# Get tree
tree = pi_client.get_tree(doc_id, node_summary=True)["result"]

# Chat API (streaming)
for chunk in pi_client.chat_completions(
    messages=[{"role": "user", "content": "your query"}],
    doc_id=doc_id,           # single doc_id or list
    stream=True
):
    print(chunk, end='')

# Chat API (non-streaming)
response = pi_client.chat_completions(
    messages=[{"role": "user", "content": "your query"}],
    doc_id="pi-abc123def456"
)
print(response["choices"][0]["message"]["content"])
```

### 8.4 Requirements (requirements.txt — complete)

```
openai==1.101.0
pymupdf==1.26.4
PyPDF2==3.0.1
python-dotenv==1.1.0
tiktoken==0.11.0
pyyaml==6.0.2
```

### 8.5 Multi-Document Search Strategies (tutorials)

**Search by Semantics** — for diverse-topic document collections:

Scoring formula: `DocScore = (1 / sqrt(N+1)) * sum(ChunkScore(n) for n in chunks)`

Where N is the number of matching chunks per document. The `sqrt(N+1)` denominator penalizes documents with many weakly-relevant chunks, favoring fewer highly-relevant matches.

**Search by Description** — lightweight for small collections:

1. Generate one-sentence LLM description per document using the tree structure
2. For a query, use LLM to compare query against all descriptions and return doc_ids
3. Retrieve with PageIndex using the selected doc_ids

**Search by Metadata** — for structured document collections (closed beta):
1. Store doc_id + metadata in SQL table
2. Use LLM to convert query to SQL
3. Retrieve matching doc_ids
4. Run PageIndex retrieval on matches

### 8.6 Utility Functions (utils.py — key helpers)

```python
# Tree traversal
def structure_to_list(structure):
    """Returns flat list of all nodes (depth-first)"""

def get_leaf_nodes(structure):
    """Returns only leaf nodes (no children)"""

def get_nodes(structure):
    """Returns all nodes without their children (each node's own dict only)"""

# Text operations
def add_node_text(node, pdf_pages):
    """Adds 'text' field to each node from pdf_pages[(start_index, end_index)]"""

def remove_structure_text(data):
    """Removes 'text' field from all nodes"""

def add_node_text_with_labels(node, pdf_pages):
    """Adds 'text' with <physical_index_X> page labels"""

# Summary
async def generate_summaries_for_structure(structure, model=None):
    """Generates summaries for all nodes concurrently"""

# Output formatting
def format_structure(structure, order=None):
    """Reorders fields in each node dict according to order list"""
    # order example: ['title', 'node_id', 'summary', 'prefix_summary', 'text', 'line_num', 'nodes']

def remove_fields(data, fields=['text']):
    """Recursively removes specified fields from all nodes"""

def print_toc(tree, indent=0):
    """Pretty-prints tree as indented TOC"""

# JSON helpers
def extract_json(content):
    """Extracts JSON from LLM response (handles ```json blocks and raw JSON)"""
    # Also handles: None->null, trailing commas, whitespace normalization

def count_tokens(text, model=None):
    """Counts tokens using tiktoken for given model"""
```

---

## 9. Markdown Pipeline (page_index_md.py — complete)

The markdown pipeline is entirely regex-based (no LLM needed for structure extraction):

```python
def extract_nodes_from_markdown(markdown_content):
    header_pattern = r'^(#{1,6})\s+(.+)$'
    code_block_pattern = r'^```'
    node_list = []
    lines = markdown_content.split('\n')
    in_code_block = False

    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()
        if re.match(code_block_pattern, stripped_line):
            in_code_block = not in_code_block
            continue
        if not stripped_line:
            continue
        if not in_code_block:
            match = re.match(header_pattern, stripped_line)
            if match:
                title = match.group(2).strip()
                node_list.append({'node_title': title, 'line_num': line_num})
    return node_list, lines
```

### Tree Thinning (merging small nodes)

```python
def tree_thinning_for_index(node_list, min_node_token=None, model=None):
    """
    Merges nodes that have fewer tokens than min_node_token
    into their parent by absorbing all children's text.
    Works bottom-up (reverse order) so children are processed before parents.
    """
```

### Build Tree from Flat List

```python
def build_tree_from_nodes(node_list):
    """
    Converts flat list with 'level' field (1-6) into nested tree.
    Uses a stack to track parent chain.
    """
    stack = []
    root_nodes = []
    node_counter = 1

    for node in node_list:
        current_level = node['level']
        tree_node = {
            'title': node['title'],
            'node_id': str(node_counter).zfill(4),
            'text': node['text'],
            'line_num': node['line_num'],
            'nodes': []
        }
        node_counter += 1
        while stack and stack[-1][1] >= current_level:
            stack.pop()
        if not stack:
            root_nodes.append(tree_node)
        else:
            parent_node, parent_level = stack[-1]
            parent_node['nodes'].append(tree_node)
        stack.append((tree_node, current_level))
    return root_nodes
```

### Markdown Summary Generation

For nodes with fewer than `summary_token_threshold` (default 200) tokens, the node text itself is used as the summary. For larger nodes, an LLM call is made via `generate_node_summary()`.

---

## 10. Verification and Error Correction Pipeline

### 10.1 `verify_toc()` — Accuracy Sampling

```python
async def verify_toc(page_list, list_result, start_index=1, N=None, model=None):
    # If N is None, check ALL items
    # Otherwise sample N random items
    # Run check_title_appearance concurrently for all sampled items
    # Return (accuracy_float, list_of_incorrect_items)
```

**Accuracy thresholds in meta_processor:**
- `accuracy == 1.0` → return as-is
- `accuracy > 0.6` → fix incorrect entries with up to 3 retries
- `accuracy <= 0.6` → fall back to next processing mode

### 10.2 `fix_incorrect_toc_with_retries()` — Error Correction

For each incorrect entry:
1. Find the neighboring correct entries (prev/next) to bound the search range
2. Call `single_toc_item_index_fixer()` on pages within that range
3. Re-verify the fix with `check_title_appearance()`
4. Up to 3 retry attempts

### 10.3 `validate_and_truncate_physical_indices()` — Bounds Check

```python
def validate_and_truncate_physical_indices(toc_with_page_number, page_list_length, start_index=1, logger=None):
    """
    Sets physical_index to None for any entry referencing a page beyond the document.
    Prevents IndexError on truncated/corrupt PDFs.
    """
    max_allowed_page = page_list_length + start_index - 1
    for item in toc_with_page_number:
        if item.get('physical_index') is not None:
            if item['physical_index'] > max_allowed_page:
                item['physical_index'] = None
```

---

## 11. LLM API Wrappers (utils.py — complete)

```python
def ChatGPT_API(model, prompt, api_key=CHATGPT_API_KEY, chat_history=None):
    max_retries = 10
    client = openai.OpenAI(api_key=api_key)
    for i in range(max_retries):
        try:
            messages = chat_history + [{"role": "user", "content": prompt}] if chat_history else [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            time.sleep(1)
    return "Error"


def ChatGPT_API_with_finish_reason(model, prompt, api_key=CHATGPT_API_KEY, chat_history=None):
    # Same as above but returns (content, "finished" | "max_output_reached")


async def ChatGPT_API_async(model, prompt, api_key=CHATGPT_API_KEY):
    # Async version using openai.AsyncOpenAI, 10 retries with 1s sleep
```

All API calls use `temperature=0` and up to 10 retries with 1-second backoff.

---

## 12. Key Design Observations for Building an Alternative

1. **No embeddings, no vector DB**: Structure is extracted purely via LLM prompting on raw text. Retrieval is via LLM reasoning over the tree, not similarity search.

2. **Physical page tags**: Pages are wrapped with `<physical_index_N>...<physical_index_N>` (same tag for open/close). This is how the LLM maps section titles to page numbers.

3. **Dot-notation hierarchy**: The `structure` field uses `"1.2.3"` notation to express hierarchy. `list_to_tree()` converts this to nested `nodes` arrays.

4. **appear_start field**: After TOC extraction, each item gets `appear_start: yes/no` indicating if the section starts at the very top of its page. This affects how `end_index` of the previous section is computed (whether to subtract 1 or not).

5. **Three-mode fallback**: The system tries TOC-with-pages → TOC-no-pages → no-TOC. At each level, if accuracy drops below 0.6, it falls back.

6. **Recursive subdivision**: Sections exceeding 10 pages AND 20k tokens get re-processed with `process_no_toc` on just their pages, then merged back as child nodes.

7. **Chunking strategy**: `page_list_to_group_text` targets `ceil((total_tokens/N + max_tokens) / 2)` tokens per chunk with 1-page overlap. This is more conservative than a pure equal split.

8. **Continuation loops**: Long TOC outputs are continued via multi-turn chat with "please continue..." prompts until both `finish_reason == "finished"` AND completeness check returns "yes".

9. **Token counting**: Uses tiktoken with the model name for accurate token counts. Default model for tokenization is `gpt-4o-2024-11-20`.

10. **MCTS in production**: The open-source code does plain LLM tree search. The cloud API uses "LLM tree search + value function-based Monte Carlo Tree Search (MCTS)" (undisclosed details).
