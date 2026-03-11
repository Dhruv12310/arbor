"""
Prompts for TOC verification and error correction.
These are thin wrappers that delegate to tree_generation prompts — kept
separate so imports are clean.
"""

from arbor.prompts.tree_generation import (
    check_title_appearance_prompt,
    check_title_at_start_prompt,
    fix_toc_entry_prompt,
)

__all__ = [
    "check_title_appearance_prompt",
    "check_title_at_start_prompt",
    "fix_toc_entry_prompt",
]
