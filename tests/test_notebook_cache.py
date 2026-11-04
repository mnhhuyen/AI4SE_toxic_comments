"""Exercise notebook cache helpers without executing data or training cells."""

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "test.ipynb"


def helpers(tmp_path):
    notebook = json.loads(NOTEBOOK.read_text())
    source = "".join(notebook["cells"][0]["source"])
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    source_dir = tmp_path / "src" / "toxic_comments"
    source_dir.mkdir(parents=True)
    (source_dir / "model.py").write_text("VERSION = 1\n")
    namespace = dict(hashlib=hashlib, json=json, pd=pd, SRC_DIR=tmp_path / "src",
                     LABEL_COLUMNS=["label"])
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    namespace["SESSION_SOURCE_FINGERPRINT"] = namespace["source_fingerprint"]()
    return namespace


def test_fingerprint_invalidates_data_settings_and_source(tmp_path):
    ns = helpers(tmp_path)
    fingerprint = ns["experiment_fingerprint"]
    train = pd.DataFrame({"id": [1, 2], "text": ["a", "b"], "label": [0, 1]})
    validation = pd.DataFrame({"id": [3], "text": ["c"], "label": [1]})
    initial = fingerprint(train, validation, "text", max_vectors=None)
    assert initial == fingerprint(train.copy(), validation.copy(), "text", max_vectors=None)
    assert initial != fingerprint(train.iloc[::-1], validation, "text", max_vectors=None)
    assert initial != fingerprint(train, validation, "text", max_vectors=10)
    for column, value in [("id", 99), ("text", "changed"), ("label", 0)]:
        changed = validation.copy()
        changed.loc[0, column] = value
        assert initial != fingerprint(train, changed, "text", max_vectors=None)
    (tmp_path / "src" / "toxic_comments" / "model.py").write_text("VERSION = 2\n")
    with pytest.raises(RuntimeError, match="restart"):
        fingerprint(train, validation, "text", max_vectors=None)


def test_cache_only_skips_matching_successful_scores(tmp_path):
    ns = helpers(tmp_path)
    base = {"model_name": "fasttext", "pooling": "supervised_attention",
            "macro_roc_auc": 0.9, "error": None}
    rows = [base, {**base, "experiment_fingerprint": "old"},
            {**base, "experiment_fingerprint": "new", "error": "failed"},
            {**base, "experiment_fingerprint": "new", "macro_roc_auc": None}]
    assert ns["completed_configurations_for"](rows, "new") == set()
    rows.append({**base, "experiment_fingerprint": "new"})
    assert ns["completed_configurations_for"](rows, "new") == {("fasttext", "supervised_attention")}


def test_notebook_code_cells_compile():
    for index, cell in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")
