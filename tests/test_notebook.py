import json
import re

import pytest

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


def test_gitignore_cannot_hide_package_code():
    """An unanchored 'data/' rule once ignored aem_research/data/ – the whole data layer.
    Every directory rule must be anchored to the repo root or match no code folder."""
    rules = [r.strip() for r in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
             if r.strip() and not r.startswith("#")]
    caches = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
    code_dirs = {p.name for base in ("aem_research", "tests", "tools", "notebooks", "config")
                 for p in [ROOT / base, *(ROOT / base).rglob("*")]
                 if p.is_dir() and p.name not in caches and not caches & set(p.parts)}
    for rule in rules:
        if rule.endswith("/") and not rule.startswith("/"):
            assert rule.rstrip("/") not in code_dirs, f".gitignore rule {rule!r} would hide a code folder"
    assert (ROOT / "aem_research" / "data" / "fsds.py").exists()


NOTEBOOK_A = ROOT / "notebooks" / "02_paper_a_take_private_sample.ipynb"


def test_paper_a_notebook_is_valid_and_compiles():
    nb = json.loads(NOTEBOOK_A.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 5
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(codes) >= 10
    for i, cell in enumerate(codes):
        src = "".join(cell["source"])
        assert cell["outputs"] == [] and cell["execution_count"] is None
        assert not re.search(r"^\s*[!%]", src, flags=re.M), f"code cell {i} uses shell or magic syntax"
        compile(src, f"cell-{i}", "exec")
    text = NOTEBOOK_A.read_text(encoding="utf-8")
    assert "paper_a_LOCK.json" in text and "@" not in "".join("".join(c["source"]) for c in codes if "User-Agent" in "".join(c["source"]))


NOTEBOOK_D = ROOT / "notebooks" / "03_universe_a_diagnostic.ipynb"
NOTEBOOK_R = ROOT / "notebooks" / "04_results.ipynb"
NOTEBOOK_AR = ROOT / "notebooks" / "05_paper_a_results.ipynb"
NOTEBOOK_X = ROOT / "notebooks" / "06_explore.ipynb"


def test_diagnostic_notebook_is_valid_and_reads_only():
    nb = json.loads(NOTEBOOK_D.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 5
    codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(codes) >= 3
    src = "\n".join("".join(c["source"]) for c in codes)
    for i, cell in enumerate(codes):
        assert cell["outputs"] == [] and cell["execution_count"] is None
        assert not re.search(r"^\s*[!%]", "".join(cell["source"]), flags=re.M)
        compile("".join(cell["source"]), f"cell-{i}", "exec")
    # counts only: no outcome engine, no network clients, no writes to the feature caches
    assert "outcomes" not in src and "SecClient" not in src and "save_table" not in src


def test_notebooks_match_their_builders(tmp_path):
    import subprocess
    import sys
    for builder, nb in (("build_notebook.py", NOTEBOOK), ("build_notebook_paper_a.py", NOTEBOOK_A),
                        ("build_notebook_diag.py", NOTEBOOK_D), ("build_notebook_results.py", NOTEBOOK_R),
                        ("build_notebook_paper_a_results.py", NOTEBOOK_AR), ("build_notebook_explore.py", NOTEBOOK_X)):
        out = tmp_path / nb.name
        subprocess.run([sys.executable, str(ROOT / "tools" / builder), str(out)], check=True, capture_output=True)
        assert out.read_text(encoding="utf-8") == nb.read_text(encoding="utf-8"), f"{nb.name} is stale: rebuild it"


def test_every_notebook_is_stamped_and_checks_its_stamp(tmp_path):
    from aem_research import env
    for nb_path in (NOTEBOOK, NOTEBOOK_A, NOTEBOOK_D, NOTEBOOK_R, NOTEBOOK_AR, NOTEBOOK_X):
        stamp = env.notebook_build(nb_path)
        assert stamp and re.fullmatch(r"[0-9a-f]{12}", stamp), nb_path.name
        setup = "".join(next(c for c in json.loads(nb_path.read_text(encoding="utf-8"))["cells"]
                             if c["cell_type"] == "code")["source"])
        assert f'check_notebook_fresh(REPO_DIR, "{nb_path.name}", NOTEBOOK_BUILD' in setup
    # an older copy stops with a pointer to the README link; the current one passes
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / NOTEBOOK.name).write_text(NOTEBOOK.read_text(encoding="utf-8"), encoding="utf-8")
    assert env.check_notebook_fresh(tmp_path, NOTEBOOK.name, env.notebook_build(NOTEBOOK)) == env.notebook_build(NOTEBOOK)
    with pytest.raises(env.StaleNotebook, match="README link"):
        env.check_notebook_fresh(tmp_path, NOTEBOOK.name, "0" * 12)


def test_every_notebook_and_the_readme_carry_the_current_release(tmp_path):
    import aem_research
    from aem_research import env
    rel = aem_research.RELEASE
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for num, nb_path in (("01", NOTEBOOK), ("02", NOTEBOOK_A), ("03", NOTEBOOK_D), ("04", NOTEBOOK_R), ("05", NOTEBOOK_AR), ("06", NOTEBOOK_X)):
        assert env.notebook_release(nb_path) == rel, nb_path.name
        cells = json.loads(nb_path.read_text(encoding="utf-8"))["cells"]
        title = "".join(cells[0]["source"]).splitlines()[0]
        assert title.startswith("# ") and title.endswith(f"· v{rel}"), title
        assert f"[Open notebook {num} · v{rel} in Colab]" in readme
    # an older copy names its own version and the one to open
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / NOTEBOOK.name).write_text(NOTEBOOK.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(env.StaleNotebook, match=rf"This is v{rel - 1} \(build 0+\).*Open v{rel} from the README link"):
        env.check_notebook_fresh(tmp_path, NOTEBOOK.name, "0" * 12, release=rel - 1)


def test_paste_blocks_lead_with_the_release():
    from aem_research.report import paper_a_block, paste_block
    assert paste_block({"release": 7}).splitlines()[0].startswith("AEM RUN SUMMARY v7 – ")
    assert paste_block({"release": 7}).splitlines()[2].startswith("v7 · run ")
    assert paper_a_block({"release": 7}).splitlines()[0].startswith("PAPER A RUN SUMMARY v7 – ")
    assert paste_block({}).splitlines()[0].startswith("AEM RUN SUMMARY v? – ")


def test_results_notebook_is_valid_and_computes_nothing_before_its_checks():
    nb = json.loads(NOTEBOOK_R.read_text(encoding="utf-8"))
    codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, cell in enumerate(codes):
        assert cell["outputs"] == [] and cell["execution_count"] is None
        assert not re.search(r"^\s*[!%]", "".join(cell["source"]), flags=re.M)
        compile("".join(cell["source"]), f"cell-{i}", "exec")
    checks, results = "".join(codes[1]["source"]), "".join(codes[2]["source"])
    assert "data_run_problems(cfg, RUN)" in checks and "failed_gates(RUN)" in checks
    assert results.startswith("# ── Outcomes") and "if not problems:" in results.splitlines()[2]
    assert "SecClient" not in "".join("".join(c["source"]) for c in codes)        # no downloads


def test_paper_a_results_notebook_is_valid_and_computes_nothing_before_its_checks():
    nb = json.loads(NOTEBOOK_AR.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 5
    codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for i, src in enumerate(codes):
        assert not re.search(r"^\s*[!%]", src, flags=re.M)
        compile(src, f"cell-{i}", "exec")
    checks = next(i for i, src in enumerate(codes) if "data_run_problems(cfg_a, RUN)" in src)
    compute = next(i for i, src in enumerate(codes) if "par.confirmatory(" in src)
    assert checks < compute and "if not problems:" in codes[compute] and "paper_a_LOCK.json" in codes[checks]
    assert "SecClient" not in "\n".join(codes) and "requests" not in "\n".join(codes)        # nothing is downloaded


def test_exploration_notebook_needs_both_locks_and_downloads_nothing():
    nb = json.loads(NOTEBOOK_X.read_text(encoding="utf-8"))
    codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for i, src in enumerate(codes):
        assert not re.search(r"^\s*[!%]", src, flags=re.M)
        compile(src, f"cell-{i}", "exec")
    checks = next(i for i, src in enumerate(codes) if "data_run_problems(cfg_b, RUN_B)" in src)
    tables = next(i for i, src in enumerate(codes) if "px.run_all(" in src)
    assert checks < tables and "data_run_problems(cfg_a, RUN_A)" in codes[checks] and "if not problems:" in codes[tables]
    assert "SecClient" not in "\n".join(codes) and "requests" not in "\n".join(codes)
