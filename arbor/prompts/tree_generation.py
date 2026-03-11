"""
All prompts for tree generation — exact ports of PageIndex's 14 prompts.

Each function returns a ready-to-send prompt string.
The prompts are kept as close to the originals as possible — same instructions,
same JSON output formats, same wording. This ensures the same LLMs that work
with PageIndex will work with Arbor without re-tuning.
"""

from __future__ import annotations

import json


def toc_detector_prompt(content: str) -> str:
    """
    Detect whether the given text contains a table of contents.

    Returns JSON: {"thinking": str, "toc_detected": "yes" | "no"}
    """
    return f"""Your job is to detect if there is a table of content provided in the given text.

Given text: {content}

return the following JSON format:
{{
    "thinking": <why do you think there is a table of content in the given text>
    "toc_detected": "<yes or no>",
}}

Directly return the final JSON structure. Do not output anything else.
Please note: abstract,summary, notation list, figure list, table list, etc. are not table of contents."""


def check_toc_complete_prompt(content: str, toc: str) -> str:
    """
    Check whether the extracted TOC content is complete (covers all main sections).

    Returns JSON: {"thinking": str, "completed": "yes" | "no"}
    """
    return f"""You are given a partial document and a table of contents.
Your job is to check if the table of contents is complete, which it contains all the main sections in the partial document.

Reply format:
{{
    "thinking": <why do you think the table of contents is complete or not>
    "completed": "yes" or "no"
}}
Directly return the final JSON structure. Do not output anything else.

Document:
{content}

Table of contents:
{toc}"""


def check_toc_transform_complete_prompt(content: str, toc: str) -> str:
    """
    Check whether a TOC transformation (raw → JSON) is complete.

    Returns JSON: {"thinking": str, "completed": "yes" | "no"}
    """
    return f"""You are given a raw table of contents and a table of contents.
Your job is to check if the table of contents is complete.

Reply format:
{{
    "thinking": <why do you think the cleaned table of contents is complete or not>
    "completed": "yes" or "no"
}}
Directly return the final JSON structure. Do not output anything else.

Raw Table of contents:
{content}

Cleaned Table of contents:
{toc}"""


def extract_toc_prompt(content: str) -> str:
    """
    Extract the raw TOC text from the given document text.

    Returns: raw TOC text (not JSON).
    """
    return f"""Your job is to extract the full table of contents from the given text, replace ... with :

Given text: {content}

Directly return the full table of contents content. Do not output anything else."""


def detect_page_index_prompt(toc_content: str) -> str:
    """
    Detect whether the TOC contains page numbers.

    Returns JSON: {"thinking": str, "page_index_given_in_toc": "yes" | "no"}
    """
    return f"""You will be given a table of contents.

Your job is to detect if there are page numbers/indices given within the table of contents.

Given text: {toc_content}

Reply format:
{{
    "thinking": <why do you think there are page numbers/indices given within the table of contents>
    "page_index_given_in_toc": "<yes or no>"
}}
Directly return the final JSON structure. Do not output anything else."""


def toc_transformer_prompt(toc_content: str) -> str:
    """
    Transform raw TOC text into structured JSON.

    structure is the dot-notation hierarchy index: "1", "1.2", "1.2.3", etc.

    Returns JSON:
    {
        "table_of_contents": [
            {"structure": str, "title": str, "page": int | null},
            ...
        ]
    }
    """
    return f"""You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

structure is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

The response should be in the following JSON format:
{{
table_of_contents: [
    {{
        "structure": <structure index, "x.x.x" or None> (string),
        "title": <title of the section>,
        "page": <page number or None>,
    }},
    ...
    ],
}}
You should transform the full table of contents in one go.
Directly return the final JSON structure, do not output anything else.

Given table of contents:
{toc_content}"""


def toc_transformer_continue_prompt() -> str:
    """
    Continuation prompt for truncated toc_transformer output.

    Used when the LLM hits its output limit mid-generation.
    """
    return "please continue the generation of table of contents , directly output the remaining part of the structure"


def toc_index_extractor_prompt(toc: list[dict] | str, content: str) -> str:
    """
    Map TOC items to their physical page numbers.

    content is a block of tagged pages containing <physical_index_N> markers.
    LLM fills in the physical_index for each TOC item found in the provided pages.

    Returns JSON: list of {structure, title, physical_index}
    """
    toc_str = json.dumps(toc, indent=2) if not isinstance(toc, str) else toc
    return f"""You are given a table of contents in a json format and several pages of a document, your job is to add the physical_index to the table of contents in the json format.

The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

The response should be in the following JSON format:
[
    {{
        "structure": <structure index, "x.x.x" or None> (string),
        "title": <title of the section>,
        "physical_index": "<physical_index_X>" (keep the format)
    }},
    ...
]

Only add the physical_index to the sections that are in the provided pages.
If the section is not in the provided pages, do not add the physical_index to it.
Directly return the final JSON structure. Do not output anything else.

Table of contents:
{toc_str}

Document pages:
{content}"""


def add_page_number_prompt(part: str, structure: list[dict] | str) -> str:
    """
    Fill in physical page numbers for a TOC whose entries lack page numbers.

    Used in TOC_NO_PAGES mode: processes the document sequentially to find
    where each section starts.

    Returns JSON: list of {structure, title, start, physical_index}
    """
    structure_str = json.dumps(structure, indent=2) if not isinstance(structure, str) else structure
    return f"""You are given an JSON structure of a document and a partial part of the document. Your task is to check if the title that is described in the structure is started in the partial given document.

The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

If the full target section starts in the partial given document, insert the given JSON structure with the "start": "yes", and "start_index": "<physical_index_X>".

If the full target section does not start in the partial given document, insert "start": "no",  "start_index": None.

The response should be in the following format.
    [
        {{
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "start": "<yes or no>",
            "physical_index": "<physical_index_X> (keep the format)" or None
        }},
        ...
    ]
The given structure contains the result of the previous part, you need to fill the result of the current part, do not change the previous result.
Directly return the final JSON structure. Do not output anything else.

Current Partial Document:
{part}

Given Structure:
{structure_str}"""


def generate_toc_init_prompt(part: str) -> str:
    """
    Generate a tree structure from the FIRST chunk of a document with no TOC.

    Returns JSON: list of {structure, title, physical_index}
    """
    return f"""You are an expert in extracting hierarchical tree structure, your task is to generate the tree structure of the document.

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


Directly return the final JSON structure. Do not output anything else.

Given text:
{part}"""


def generate_toc_continue_prompt(toc_content: list[dict] | str, part: str) -> str:
    """
    Continue generating a tree structure from a SUBSEQUENT chunk.

    Receives the TOC built so far and adds entries for the current chunk.

    Returns JSON: list of NEW {structure, title, physical_index} items to append.
    """
    toc_str = json.dumps(toc_content, indent=2) if not isinstance(toc_content, str) else toc_content
    return f"""You are an expert in extracting hierarchical tree structure.
You are given a tree structure of the previous part and the text of the current part.
Your task is to continue the tree structure from the previous part to include the current part.

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
        ...
    ]

Directly return the additional part of the final JSON structure. Do not output anything else.

Given text:
{part}

Previous tree structure:
{toc_str}"""


def check_title_appearance_prompt(title: str, page_text: str) -> str:
    """
    Verify that a section title appears in the given page text.

    Returns JSON: {"thinking": str, "answer": "yes" | "no"}
    """
    return f"""Your job is to check if the given section appears or starts in the given page_text.

Note: do fuzzy matching, ignore any space inconsistency in the page_text.

The given section title is {title}.
The given page_text is {page_text}.

Reply format:
{{
    "thinking": <why do you think the section appears or starts in the page_text>
    "answer": "yes or no" (yes if the section appears or starts in the page_text, no otherwise)
}}
Directly return the final JSON structure. Do not output anything else."""


def check_title_at_start_prompt(title: str, page_text: str) -> str:
    """
    Check whether a section title appears at the BEGINNING of the page (no prior content).

    Returns JSON: {"thinking": str, "start_begin": "yes" | "no"}
    """
    return f"""You will be given the current section title and the current page_text.
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


def fix_toc_entry_prompt(section_title: str, content: str) -> str:
    """
    Find the correct physical page number for a TOC entry that was mapped incorrectly.

    content is a window of tagged pages around the expected location.

    Returns JSON: {"thinking": str, "physical_index": "<physical_index_X>"}
    """
    return f"""You are given a section title and several pages of a document, your job is to find the physical index of the start page of the section in the partial document.

The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

Reply in a JSON format:
{{
    "thinking": <explain which page, started and closed by <physical_index_X>, contains the start of this section>,
    "physical_index": "<physical_index_X>" (keep the format)
}}
Directly return the final JSON structure. Do not output anything else.

Section Title:
{section_title}

Document pages:
{content}"""


def generate_summary_prompt(text: str) -> str:
    """
    Generate a concise summary of a document section.

    Returns: plain text summary (no JSON).
    """
    return f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

Partial Document Text: {text}

Directly return the description, do not include any other text."""


def generate_doc_description_prompt(structure: dict | str) -> str:
    """
    Generate a one-sentence description of the entire document.

    structure is the document's tree (without text fields) as JSON.

    Returns: plain text one-sentence description.
    """
    structure_str = json.dumps(structure, indent=2) if not isinstance(structure, str) else structure
    return f"""Your are an expert in generating descriptions for a document.
You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.

Document Structure: {structure_str}

Directly return the description, do not include any other text."""
