from __future__ import annotations
import os
import pandas as pd
from typing import Tuple
from ..config import PipelineConfig
from .normalize import normalize_aspect, normalize_node, infer_node, normalize_blocks, blocks_count

def run_energy_pipeline(cfg: PipelineConfig, excel_path: str, out_dir: str) -> Tuple[str, str]:
    e = cfg.energy

    df_expr = pd.read_excel(excel_path, sheet_name=e.expressions_sheet)
    df_fw   = pd.read_excel(excel_path, sheet_name=e.framework_sheet)

    df_expr.columns = [str(c).strip() for c in df_expr.columns]
    df_fw.columns   = [str(c).strip() for c in df_fw.columns]

    missing_expr = [c for c in e.required_expr_cols if c not in df_expr.columns]
    if missing_expr:
        raise ValueError(f"Missing in ExpressionsMapping: {missing_expr}")
    if e.framework_key_col not in df_fw.columns:
        raise ValueError(f"Missing in Framework: [{e.framework_key_col}]")

    # Normalize columns
    asp_col = "Aspects of Woman Track"
    node_col = "energy_node/energy block behind it/ inner block"
    prob_col = "Problem statement"
    dual_col = "Duality Check"
    blocks_col = "deeper_blocks/ pshychlogical issues"

    df = df_expr.copy()
    df[asp_col]  = df[asp_col].apply(lambda x: normalize_aspect(x, e.aspects_allowed))
    df[node_col] = df[node_col].apply(lambda x: normalize_node(x, e.nodes_allowed))
    df[blocks_col] = df[blocks_col].apply(normalize_blocks)

    # Fill blank nodes
    blank = df[node_col].fillna("").astype(str).str.strip().eq("")
    df.loc[blank, node_col] = df.loc[blank].apply(
        lambda r: infer_node(str(r.get(prob_col, "")), str(r.get(blocks_col, ""))),
        axis=1
    )

    # Enrich with framework
    fw = df_fw.copy()
    fw[e.framework_key_col] = fw[e.framework_key_col].astype(str).str.strip().str.lower()
    fw_lookup = fw.set_index(e.framework_key_col).to_dict(orient="index")

    for col in e.framework_cols:
        if col not in df.columns:
            df[col] = ""

    for i, r in df.iterrows():
        node = str(r.get(node_col, "")).strip().lower()
        ref = fw_lookup.get(node, {})
        for col in e.framework_cols:
            val = ref.get(col, "")
            if pd.notna(val):
                df.at[i, col] = val

    # Quality gate
    def nonempty(x): return str(x).strip() if pd.notna(x) else ""

    df["_prob_len"]   = df[prob_col].apply(lambda x: len(nonempty(x)))
    df["_dual_len"]   = df[dual_col].apply(lambda x: len(nonempty(x)))
    df["_blocks_len"] = df[blocks_col].apply(lambda x: len(nonempty(x)))
    df["_blocks_cnt"] = df[blocks_col].apply(blocks_count)

    g = e.gates
    reject = (
        (df[asp_col] == "Unknown") |
        (~df[node_col].isin(e.nodes_allowed)) |
        (df["_prob_len"] < g.min_problem_len) |
        (df["_dual_len"] < g.min_duality_len) |
        (df["_blocks_len"] < g.min_blocks_len) |
        (df["_blocks_cnt"] < g.min_blocks_count)
    )

    gold = df[~reject].copy()
    rej  = df[reject].copy()

    os.makedirs(out_dir, exist_ok=True)
    out_gold = os.path.join(out_dir, "gold.xlsx")
    out_rej  = os.path.join(out_dir, "reject.xlsx")

    gold.to_excel(out_gold, index=False)
    rej.to_excel(out_rej, index=False)
    return out_gold, out_rej
