"""Runtime environment: Colab or local, and the folder layout on Google Drive."""
from __future__ import annotations

import os
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
