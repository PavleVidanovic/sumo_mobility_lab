"""
Pomoćne funkcije za pokretanje SUMO iz Streamlit-a: putanje, komandna linija, izlazni folderi.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


def path_for_sumo_argv(p: str | Path) -> str:
    """
    Pomoćna funkcija za formatiranje putanja u komandnoj liniji SUMO, posebno na Windowsu gde su potrebne kratke putanje bez zareza.
    """
    try:
        path = Path(p).resolve()
    except OSError:
        return str(p)
    if sys.platform != "win32":
        return str(path)
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(4096)

        def short(s: str) -> str:
            """Windows 8.3 kratka putanja (bez zareza) za SUMO argv."""
            n = ctypes.windll.kernel32.GetShortPathNameW(s, buf, len(buf))
            return buf.value if n else s

        if path.exists():
            return short(str(path))
        parent = path.parent
        if parent.exists():
            return str(Path(short(str(parent))) / path.name)
        return str(path)
    except Exception:
        return str(path)


def resolve_output_directory(out_dir: str, project_root: Path) -> Path:
    """
    Apsolutna putanja za izlazne fajlove.
    Relativna putanja = u odnosu na folder gde je app.py, ne na cwd Streamlit procesa.
    """
    raw = (out_dir or "").strip()
    if not raw or raw == ".":
        return Path.cwd().resolve()
    p = Path(raw)
    if p.is_absolute():
        return p.expanduser().resolve()
    return (project_root / p).resolve()


def output_dir_is_under_program_files(out_dir: str, project_root: Path) -> bool:
    """
    Pisanje u "Program Files" je ograničeno na Windowsu, često se javlja ako je scenario ili izlazni folder tamo (npr. netedit/OSM).
    """
    if not (out_dir or "").strip():
        return False
    try:
        resolved = resolve_output_directory(out_dir, project_root).as_posix().lower()
    except OSError:
        return False
    return "program files" in resolved


def format_sumo_cmdline(cmd: list[str]) -> str:
    """
    Jedna linija za CMD/PowerShell kopiranje; putanje sa razmacima su pravilno citirane (Windows).
    """
    if sys.platform == "win32":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def build_sumo_command(
    sumocfg: str,
    out_dir: str,
    stem: str = "run",
    use_gui: bool = False,
    fcd: bool = True,
    fcd_geo: bool = True,
    emission: bool = True,
    full_output: bool = False,
    gzip: bool = False,
) -> list[str]:
    """
    Lista argumenata za SUMO (FCD + geo, emission, opciono full) — koristi se sa subprocess bez shell-a.
    """
    cfg = Path(sumocfg)
    out = Path(out_dir)
    sfx = ".xml.gz" if gzip else ".xml"
    parts: list[str] = [
        "sumo-gui" if use_gui else "sumo",
        "-c",
        path_for_sumo_argv(cfg),
    ]

    if fcd:
        parts += ["--fcd-output", path_for_sumo_argv(out / f"{stem}_fcd{sfx}")]
        if fcd_geo:
            parts.append("--fcd-output.geo")
    if emission:
        parts += [
            "--emission-output",
            path_for_sumo_argv(out / f"{stem}_emission{sfx}"),
        ]
    if full_output:
        parts += ["--full-output", path_for_sumo_argv(out / f"{stem}_full{sfx}")]

    # CLI override: tripinfos u istom folderu kao FCD; uvek .xml (SUMO gzip za tripinfo nije uvek potreban).
    parts += ["--tripinfo-output", path_for_sumo_argv(out / f"{stem}_tripinfos.xml")]
    # Osnovni SUMO scenario iz netedit/OSM često ima stop/statistics putanje u Program Files — CLI ih prepisuje.
    parts += ["--stop-output", path_for_sumo_argv(out / f"{stem}_stopinfos.xml")]
    parts += ["--statistic-output", path_for_sumo_argv(out / f"{stem}_stats.xml")]

    return parts


def default_xml2csv_script(sumo_home: str) -> Path:
    """Uobičajena lokacija xml2csv.py u SUMO instalaciji."""
    return Path(sumo_home) / "tools" / "xml" / "xml2csv.py"
