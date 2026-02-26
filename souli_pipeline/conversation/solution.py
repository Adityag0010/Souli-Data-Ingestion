"""
Solution retriever — pulls healing framework data from gold.xlsx
for the diagnosed energy node.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Column names matching energy pipeline output
NODE_COL = "energy_node/energy block behind it/ inner block"
FRAMEWORK_COLS = [
    "typical_signs",
    "primary_healing_principles",
    "primary_practices ( 7 min quick relief)",
    "deeper_meditations_program ( 7 day quick recovery)",
    "longer_program ( 7 month resilience building)",
    "Caution",
    "Health",
]


def load_framework_from_gold(gold_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load gold.xlsx and build a node → framework_solution lookup dict.
    Returns {energy_node: {col: value, ...}, ...}
    """
    df = pd.read_excel(gold_path)
    df.columns = [str(c).strip() for c in df.columns]

    if NODE_COL not in df.columns:
        logger.warning("gold.xlsx missing '%s' column", NODE_COL)
        return {}

    lookup: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        node = str(row.get(NODE_COL, "")).strip().lower()
        if not node or node in lookup:
            continue
        lookup[node] = {
            col: str(row.get(col, "")).strip()
            for col in FRAMEWORK_COLS
            if col in df.columns
        }
    return lookup


def load_framework_from_excel(excel_path: str, sheet: str = "Inner energy Framework") -> Dict[str, Dict[str, str]]:
    """
    Load directly from the raw Excel file (Inner energy Framework sheet).
    """
    df = pd.read_excel(excel_path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]

    lookup: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        node = str(row.get("energy_node", "")).strip().lower()
        if not node or node == "energy_node":
            continue
        lookup[node] = {}
        for col in df.columns:
            if col != "energy_node":
                val = str(row.get(col, "") or "").strip()
                if val and val != "nan":
                    lookup[node][col] = val
    return lookup


def get_solution_for_node(
    energy_node: str,
    framework: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    """
    Get the framework solution dict for a given energy node.
    Returns empty dict if not found.
    """
    node = (energy_node or "").strip().lower()
    solution = framework.get(node, {})
    if not solution:
        # Try partial match
        for key in framework:
            if node in key or key in node:
                solution = framework[key]
                break
    return solution


def format_solution_text(energy_node: str, solution: Dict[str, str]) -> str:
    """
    Format the solution as a readable text block for display or TTS.
    """
    node_label = energy_node.replace("_", " ").title()
    lines = [f"Based on your {node_label} state, here's what can help:\n"]

    healing = solution.get("primary_healing_principles", "")
    if healing:
        lines.append(f"Core principles:\n{healing.strip()}\n")

    practices = solution.get("primary_practices ( 7 min quick relief)", "")
    if practices:
        lines.append(f"Quick relief (7 min):\n{practices.strip()}\n")

    deeper = solution.get("deeper_meditations_program ( 7 day quick recovery)", "")
    if deeper:
        lines.append(f"7-day recovery:\n{deeper.strip()}\n")

    caution = solution.get("Caution", "")
    if caution:
        lines.append(f"Note: {caution.strip()}\n")

    return "\n".join(lines)
