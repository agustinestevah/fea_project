"""
llm_calls package — LLM evaluation utilities for the FEA project.

Automatically ensures the project root is on sys.path so that
`free_entailments_algorithm_utils` and sibling modules are importable
from any working directory (e.g. when papermill launches a fresh kernel).
"""
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
