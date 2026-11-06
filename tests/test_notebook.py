import json
import re

from conftest import ROOT

NOTEBOOK = ROOT / "notebooks" / "01_aem_base_rate_skeleton.ipynb"


def test_notebook_is_valid_and_compiles():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 5
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids)) and all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", i) for i in ids)
    codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(codes) >= 8
    for i, cell in enumerate(codes):
        src = "".join(cell["source"])
        assert cell["outputs"] == [] and cell["execution_count"] is None
        assert not re.search(r"^\s*[!%]", src, flags=re.M), f"code cell {i} uses shell or magic syntax"
        compile(src, f"cell-{i}", "exec")


def test_no_vendor_data_or_secrets_in_repo():
    banned = (".parquet", ".pkl", ".zip", ".gz")
    files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts]
    assert not [p for p in files if p.name.lower().endswith(banned)]
    pattern = re.compile(r"(api[_-]?key|token)\s*=\s*['\"][A-Za-z0-9]{16,}", re.I)
    for p in files:
        if p.suffix in (".py", ".ipynb", ".yaml", ".md"):
            assert not pattern.search(p.read_text(encoding="utf-8", errors="ignore")), p
