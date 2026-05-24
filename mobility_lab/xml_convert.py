"""
Pokretanje SUMO alata: "xml2csv" i generički "sumo" subprocess (bez shell-a).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from mobility_lab.sumo_cli import path_for_sumo_argv


def run_xml2csv(
    xml2csv_script: Path,
    xml_input: Path,
    *,
    python_exe: Path | None = None,
    output_csv: Path | None = None,
    separator: str = ";",
) -> tuple[int, str, str]:
    """
    Pokreće SUMO `tools/xml/xml2csv.py`.
    Vraća (returncode, stdout, stderr).
    """
    py = Path(python_exe) if python_exe else Path(sys.executable)
    cmd: list[str] = [
        str(py),
        str(xml2csv_script),
        str(xml_input),
        "--separator",
        separator,
    ]
    if output_csv is not None:
        cmd += ["--output", str(output_csv)]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_sumo(
    cmd: list[str], *, cwd: Path | None = None, timeout_s: int | None = None
) -> tuple[int, str, str]:
    """
    Pokreće SUMO bez shell-a — putanje sa razmacima i (x86) rade ispravno.
    """
    proc = subprocess.run(
        cmd,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=path_for_sumo_argv(cwd) if cwd else None,
        timeout=timeout_s,
    )
    return proc.returncode, proc.stdout, proc.stderr
