"""Step 3 tests: All prompts render correctly."""

import json
import pytest

from arbor.prompts.tree_generation import (
    toc_detector_prompt,
    check_toc_complete_prompt,
    check_toc_transform_complete_prompt,
    extract_toc_prompt,
    detect_page_index_prompt,
    toc_transformer_prompt,
    toc_index_extractor_prompt,
    add_page_number_prompt,
    generate_toc_init_prompt,
    generate_toc_continue_prompt,
    check_title_appearance_prompt,
    check_title_at_start_prompt,
    fix_toc_entry_prompt,
    generate_summary_prompt,
    generate_doc_description_prompt,
    toc_transformer_continue_prompt,
)
from arbor.prompts.tree_search import (
    tree_search_prompt,
    tree_search_with_preference_prompt,
    answer_generation_prompt,
)


class TestTreeGenerationPrompts:
    def test_toc_detector_contains_content(self):
        p = toc_detector_prompt("Chapter 1 ... 5\nChapter 2 ... 10")
        assert "Chapter 1" in p
        assert "toc_detected" in p
        assert "yes or no" in p

    def test_check_toc_complete(self):
        p = check_toc_complete_prompt("doc text", "1. Intro\n2. Body")
        assert "doc text" in p
        assert "1. Intro" in p
        assert "completed" in p

    def test_check_toc_transform_complete(self):
        p = check_toc_transform_complete_prompt("raw toc", "json toc")
        assert "raw toc" in p
        assert "json toc" in p

    def test_extract_toc(self):
        p = extract_toc_prompt("Contents\n1. Intro .. 5")
        assert "Contents" in p
        assert "table of contents" in p.lower()

    def test_detect_page_index(self):
        p = detect_page_index_prompt("1. Intro ... 5")
        assert "1. Intro" in p
        assert "page_index_given_in_toc" in p

    def test_toc_transformer(self):
        p = toc_transformer_prompt("1. Intro\n2. Methods\n3. Results")
        assert "table_of_contents" in p
        assert "structure" in p
        assert "x.x.x" in p

    def test_toc_transformer_continue(self):
        p = toc_transformer_continue_prompt()
        assert "continue" in p.lower()

    def test_toc_index_extractor_with_list(self):
        toc = [{"structure": "1", "title": "Intro"}]
        p = toc_index_extractor_prompt(toc, "<physical_index_1>\nText\n<physical_index_1>")
        assert "Intro" in p
        assert "physical_index" in p

    def test_toc_index_extractor_with_string(self):
        p = toc_index_extractor_prompt("raw toc string", "page content")
        assert "raw toc string" in p

    def test_add_page_number(self):
        structure = [{"structure": "1", "title": "Intro"}]
        p = add_page_number_prompt("<physical_index_1>\ntext\n<physical_index_1>", structure)
        assert "Intro" in p
        assert "start_index" in p

    def test_generate_toc_init(self):
        p = generate_toc_init_prompt("<physical_index_1>\nChapter 1\n<physical_index_1>")
        assert "Chapter 1" in p
        assert "physical_index" in p
        assert "structure" in p

    def test_generate_toc_continue(self):
        existing = [{"structure": "1", "title": "Intro", "physical_index": 1}]
        p = generate_toc_continue_prompt(existing, "<physical_index_2>\nChapter 2\n<physical_index_2>")
        assert "Intro" in p
        assert "Chapter 2" in p
        assert "continue" in p.lower()

    def test_check_title_appearance(self):
        p = check_title_appearance_prompt("Introduction", "Introduction\nThis paper...")
        assert "Introduction" in p
        assert "answer" in p
        assert "yes or no" in p

    def test_check_title_at_start(self):
        p = check_title_at_start_prompt("Methods", "Methods\nWe used...")
        assert "Methods" in p
        assert "start_begin" in p

    def test_fix_toc_entry(self):
        p = fix_toc_entry_prompt("Results", "<physical_index_5>\nResults\n<physical_index_5>")
        assert "Results" in p
        assert "physical_index" in p
        assert "thinking" in p

    def test_generate_summary(self):
        p = generate_summary_prompt("This is a long section about machine learning...")
        assert "machine learning" in p
        assert "description" in p.lower()

    def test_generate_doc_description_with_dict(self):
        structure = {"structure": [{"title": "Intro"}]}
        p = generate_doc_description_prompt(structure)
        assert "Intro" in p
        assert "one-sentence" in p

    def test_generate_doc_description_with_string(self):
        p = generate_doc_description_prompt("plain string structure")
        assert "plain string structure" in p


class TestTreeSearchPrompts:
    def test_tree_search_basic(self):
        tree = [{"node_id": "0001", "title": "Chapter 1", "summary": "About X"}]
        p = tree_search_prompt("What is X?", tree)
        assert "What is X?" in p
        assert "node_list" in p
        assert "thinking" in p

    def test_tree_search_with_preference(self):
        tree = [{"node_id": "0001", "title": "Financials"}]
        p = tree_search_with_preference_prompt("What is EBITDA?", tree, "Check MD&A section")
        assert "EBITDA" in p
        assert "MD&A" in p
        assert "node_list" in p

    def test_answer_generation(self):
        p = answer_generation_prompt("What is backprop?", "Backprop is an algorithm...")
        assert "What is backprop?" in p
        assert "Backprop is an algorithm" in p
        assert "answer" in p.lower()

    def test_tree_search_with_json_dict(self):
        tree = {"nodes": [{"node_id": "0001", "title": "Intro"}]}
        p = tree_search_prompt("What is the intro about?", tree)
        assert "0001" in p

    def test_tree_search_with_string(self):
        p = tree_search_prompt("test query", '{"nodes": []}')
        assert "test query" in p
