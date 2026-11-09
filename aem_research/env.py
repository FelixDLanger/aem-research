"""Runtime environment: Colab or local, and the folder layout on Google Drive."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def colab_secret(name: str) -> str | None:
    """Value of a Colab secret, or None outside Colab or when the secret is missing."""
    try:
        from google.colab import userdata  # type: ignore
        value = userdata.get(name)
        return value or None
    except Exception:
        return None


@dataclass(frozen=True)
class Paths:
    drive_root: Path
    work: Path
    stooq_dir: Path
    sec_dir: Path
    cache: Path
    output: Path
    simfin_dirs: tuple[Path, ...]

    def ensure(self) -> "Paths":
        for p in (self.work, self.stooq_dir, self.sec_dir, self.cache, self.output):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def describe(self) -> str:
        lines = [f"Drive root: {self.drive_root}", f"Study folder: {self.work}"]
        lines += [f"SimFin cache candidate: {p}  ({'found' if p.exists() else 'missing'})"
                  for p in self.simfin_dirs]
        lines.append(f"Stooq folder: {self.stooq_dir}")
        return "\n".join(lines)


def resolve_paths(drive_root: str | os.PathLike | None = None,
                  simfin_dirs: list[str] | tuple[str, ...] | None = None) -> Paths:
    root = Path(drive_root or os.environ.get("AEM_DRIVE_ROOT", "/content/drive/MyDrive"))
    work = root / "aem-research"
    rel = simfin_dirs or ("cvm-research/data/raw", "cvm-research/data")
    return Paths(
        drive_root=root,
        work=work,
        stooq_dir=work / "data" / "stooq",
        sec_dir=work / "data" / "sec",
        cache=work / "cache",
        output=work / "output",
        simfin_dirs=tuple(root / r for r in rel),
    )


class StaleNotebook(RuntimeError):
    """The notebook being run is an older copy than the repository's."""


def notebook_build(path) -> str | None:
    """The build stamp a generated notebook carries in its setup cell (NOTEBOOK_BUILD = "...")."""
    try:
        nb = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    for cell in nb.get("cells", []):
        for line in cell.get("source", []):
            m = re.match(r'\s*NOTEBOOK_BUILD = "([0-9a-f]+)"', line)
            if m:
                return m.group(1)
    return None


def check_notebook_fresh(repo_dir, notebook_name: str, build: str, *, colab_url: str | None = None) -> str | None:
    """Stop when the notebook being run is not the repository's current one.

    Colab keeps an opened notebook as it was; the setup cell pulls the newest code, so an old
    copy would run new code with old steps – and finish without an error. The builders stamp
    every notebook with a hash of its cells; this compares the running copy's stamp with the
    one in the freshly pulled repository. Returns the repository's stamp."""
    repo_build = notebook_build(Path(repo_dir) / "notebooks" / notebook_name)
    if repo_build and build and repo_build != build:
        where = f": {colab_url}" if colab_url else ""
        raise StaleNotebook(f"This is an older copy of {notebook_name} (build {build}; the repository's is "
                            f"{repo_build}). Open it again from the README link{where} – its steps changed.")
    return repo_build
