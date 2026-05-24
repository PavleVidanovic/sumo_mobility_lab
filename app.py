"""
SUMO Mobility Lab — Streamlit aplikacija.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mobility_lab.analysis import (
    aggregates_by_vehicle_type,
    df_to_csv_bytes,
    matplotlib_co2_timeseries,
    matplotlib_co2_timeseries_by_type,
    matplotlib_co2_total_by_type,
    matplotlib_jump_hist,
    matplotlib_speed_vs_co2,
    matplotlib_traj_length_hist,
    skmob_jump_and_rog,
    trajectory_lengths_movingpandas,
)
from mobility_lab.folium_map import (
    ensure_lat_lon_columns,
    mobility_map_trajectory_and_heatmap,
)
from mobility_lab.time_animation import (
    folium_time_animation_map,
    prepare_time_animation,
    prepared_json_size_kb,
)
from mobility_lab.io_sumo import (
    apply_vehicle_name_cleanup,
    merge_fcd_emission,
    normalize_emission_like,
    normalize_fcd_like,
    read_sumo_csv,
    short_vehicle_id,
)
from mobility_lab.sumo_cli import (
    build_sumo_command,
    default_xml2csv_script,
    format_sumo_cmdline,
    output_dir_is_under_program_files,
    resolve_output_directory,
)
from mobility_lab.insights import (
    build_vehicle_profile,
    insights_narrative,
    matplotlib_metrics_by_rog_tertile,
    matplotlib_rog_vs_length,
    top_wide_and_dirty,
)
from mobility_lab.mobiml_pipeline import (
    folium_h3_hotspots_map,
    run_pipeline as run_mobiml_pipeline,
)
from mobility_lab.xml_convert import run_sumo, run_xml2csv

SUMO_HOME_DEFAULT = r"C:\Program Files (x86)\Eclipse\Sumo"
DEFAULT_XML2CSV_SCRIPT = str(default_xml2csv_script(SUMO_HOME_DEFAULT).resolve())

try:
    from streamlit_folium import st_folium
except ImportError:
    st_folium = None


def _default_scenario_dir() -> Path:
    newer = ROOT / "my_simulations" / "2026-05-22-11-15-56"
    if newer.is_dir():
        return newer
    return ROOT / "my_simulations" / "2026-04-12-21-02-35"


def _default_sumocfg() -> str:
    p = _default_scenario_dir() / "osm.sumocfg"
    return str(p.resolve()) if p.is_file() else ""


def _default_edge_data() -> str:
    p = _default_scenario_dir() / "edgeData.xml"
    return str(p.resolve()) if p.is_file() else ""


def _default_net_xml() -> str:
    for name in ("osm.net.xml.gz", "osm.net.xml"):
        p = _default_scenario_dir() / name
        if p.is_file():
            return str(p.resolve())
    return ""


def _sumo_output_dir(out_dir: str) -> Path:
    if not (out_dir or "").strip():
        return (ROOT / "sumo_output").resolve()
    return resolve_output_directory(out_dir, ROOT)


def _default_fcd_csv() -> str:
    for base in (_default_scenario_dir() / "output", ROOT / "sumo_output"):
        p = base / "run_fcd.csv"
        if p.is_file():
            return str(p.resolve())
    return ""


def _default_emission_csv() -> str:
    for base in (_default_scenario_dir() / "output", ROOT / "sumo_output"):
        p = base / "run_emission.csv"
        if p.is_file():
            return str(p.resolve())
    return ""


def _default_xml_in_fcd() -> str:
    for base in (_default_scenario_dir() / "output", ROOT / "sumo_output"):
        p = base / "run_fcd.xml"
        if p.is_file():
            return str(p.resolve())
    return ""


def _init_session() -> None:
    keys = {
        "fcd_path": "",
        "em_path": "",
        "fcd_raw": None,
        "em_raw": None,
        "fcd_n": None,
        "is_geo": False,
        "merged": None,
        "lengths": None,
        "jump": None,
        "rog": None,
        "analysis_notes": "",
        "mobiml_result": None,
        "anim_prep": None,
        "flow_step": 0,
        "entry_choice": None,
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _pyplot_in_column(col, fig: plt.Figure | None) -> None:
    if fig is None:
        return
    fig.set_size_inches(4.6, 3.2)
    try:
        fig.tight_layout()
    except Exception:
        pass
    col.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _pyplot_wide(fig: plt.Figure | None) -> None:
    if fig is None:
        return
    try:
        fig.tight_layout()
    except Exception:
        pass
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


_init_session()

st.set_page_config(
    page_title="SUMO Mobility Lab", layout="wide", initial_sidebar_state="collapsed"
)

# --- Korak 0: izbor toka ---
if int(st.session_state.get("flow_step") or 0) == 0:
    st.title("SUMO Mobility Lab")
    st.caption(
        "SUMO → XML → CSV → analiza trajektorija (MovingPandas / scikit-mobility) → mapa"
    )
    st.markdown("### Korak 1 — kako nastavljaš?")
    c0a, c0b = st.columns(2)
    with c0a:
        st.markdown(
            "**Generisanje** — pokreni SUMO u aplikaciji, zatim konvertuj XML u CSV."
        )
        if st.button("Generisanje", type="primary", use_container_width=True):
            st.session_state["entry_choice"] = "generate"
            st.session_state["flow_step"] = 1
            st.rerun()
    with c0b:
        st.markdown(
            "**Već imam generisane podatke** — preskoči na učitavanje CSV i analizu."
        )
        if st.button(
            "Već imam generisane podatke", type="secondary", use_container_width=True
        ):
            st.session_state["entry_choice"] = "have_csv"
            st.session_state["flow_step"] = 2
            st.rerun()
    st.stop()

st.title("SUMO Mobility Lab")
choice = st.session_state.get("entry_choice")
flow = int(st.session_state.get("flow_step") or 0)

with st.sidebar:
    st.caption("Navigacija")
    if st.button("Početak (izbor načina rada)"):
        st.session_state["flow_step"] = 0
        st.session_state["entry_choice"] = None
        st.rerun()
    if flow == 2 and choice == "generate":
        if st.button("Nazad na generisanje"):
            st.session_state["flow_step"] = 1
            st.rerun()

# --- Korak 1: SUMO + xml2csv ---
if flow == 1 and choice == "generate":
    st.header("Korak 1 — generisanje podataka")
    st.markdown("Pokreni simulaciju, zatim konvertuj potrebne XML fajlove u CSV.")

    st.subheader("SUMO — FCD / emission / ostalo")
    st.caption(
        "**Run SUMO** šalje istu komandu kao ispod (subprocess). "
        "Podrazumevano: scenario iz `my_simulations/…`, izlaz u `sumo_output/`."
    )
    c1, c2 = st.columns(2)
    with c1:
        sumocfg = st.text_input(
            "Putanja do *.sumocfg",
            value=_default_sumocfg(),
            placeholder=r"C:\...\osm.sumocfg",
            key="s1_sumocfg",
        )
        out_dir = st.text_input(
            "Izlazni folder",
            value=str((ROOT / "sumo_output").resolve()),
            key="s1_outdir",
        )
        stem = st.text_input("Prefiks fajlova", value="run", key="s1_stem")
    with c2:
        use_gui = st.checkbox("sumo-gui (umesto sumo)", value=False, key="s1_gui")
        fcd = st.checkbox("FCD izlaz", value=True, key="s1_fcd")
        fcd_geo = st.checkbox("--fcd-output.geo (WGS84)", value=True, key="s1_geo")
        emission = st.checkbox("Emission izlaz", value=True, key="s1_em")
        full_out = st.checkbox("Full izlaz", value=False, key="s1_full")
        gzip = st.checkbox("Gzip (.xml.gz)", value=False, key="s1_gzip")

    out_for_cmd = _sumo_output_dir(out_dir)
    cmd = build_sumo_command(
        sumocfg or "osm.sumocfg",
        str(out_for_cmd),
        stem=stem or "run",
        use_gui=use_gui,
        fcd=fcd,
        fcd_geo=fcd_geo,
        emission=emission,
        full_output=full_out,
        gzip=gzip,
    )
    st.code(format_sumo_cmdline(cmd), language="powershell")

    if st.button("Run SUMO", type="primary", key="btn_sumo"):
        if not sumocfg or not Path(sumocfg).is_file():
            st.error("Nevažeća putanja do sumocfg.")
        elif output_dir_is_under_program_files(out_dir, ROOT):
            st.error(
                "Izlazni folder je ispod **Program Files** — obično nije dozvoljeno pisanje. "
                "Koristi npr. folder na Desktopu ili `sumo_output` u projektu."
            )
        else:
            cwd = Path(sumocfg).parent
            out_abs = _sumo_output_dir(out_dir)
            out_abs.mkdir(parents=True, exist_ok=True)
            cmd_run = build_sumo_command(
                sumocfg,
                str(out_abs),
                stem=stem or "run",
                use_gui=use_gui,
                fcd=fcd,
                fcd_geo=fcd_geo,
                emission=emission,
                full_output=full_out,
                gzip=gzip,
            )
            with st.spinner("SUMO …"):
                try:
                    code, _out, _err = run_sumo(cmd_run, cwd=cwd, timeout_s=None)
                except FileNotFoundError:
                    st.error(
                        "`sumo` / `sumo-gui` nije na PATH-u. Dodaj SUMO `bin` u PATH ili pokreni komandu ručno."
                    )
                except subprocess.TimeoutExpired:
                    st.error("Timeout.")
                except Exception as e:
                    st.exception(e)
                else:
                    if code == 0:
                        st.success(f"SUMO završen, kod povratka: {code}")
                    else:
                        st.warning(f"SUMO kod povratka: {code}")

    st.divider()
    st.subheader("XML → CSV (`xml2csv.py`)")
    st.markdown(
        "Dokumentacija: [xml2csv.py](https://sumo.dlr.de/docs/Tools/Xml.html#xml2csvpy) — separator `;` kao u SUMO izvozu."
    )
    sumo_home = st.text_input(
        "SUMO HOME (folder instalacije)",
        value=SUMO_HOME_DEFAULT,
        key="s1_sumo_home",
    )
    xml2csv = st.text_input(
        "Putanja do xml2csv.py",
        value=DEFAULT_XML2CSV_SCRIPT,
        key="s1_xml2csv",
    )
    xml_in = st.text_input(
        "Ulazni XML",
        value=_default_xml_in_fcd(),
        placeholder=r"C:\...\run_fcd.xml",
        key="s1_xml_in",
    )
    csv_out = st.text_input("Izlazni CSV (opciono)", value="", key="s1_csv_out")
    st.caption("Jedan XML po pokretanju — FCD i emission konvertuj odvojeno.")
    sep = st.selectbox("Separator", [";", ","], index=0, key="s1_sep")

    if st.button("Konvertuj XML → CSV", key="btn_xml2csv"):
        sh = Path(sumo_home)
        script = Path(xml2csv) if xml2csv.strip() else default_xml2csv_script(str(sh))
        if not script.is_file():
            st.error(f"Nije pronađen xml2csv.py: {script}")
        elif not xml_in or not Path(xml_in).is_file():
            st.error("Nije pronađen ulazni XML.")
        else:
            out_path = Path(csv_out) if csv_out.strip() else None
            with st.spinner("xml2csv …"):
                code, _o, _e = run_xml2csv(
                    script, Path(xml_in), output_csv=out_path, separator=sep
                )
            if code == 0:
                st.success("Konverzija završena.")
            else:
                st.error(f"Greška, kod {code}.")

    st.divider()
    if st.button(
        "Nastavi na analizu (korak 2)", type="primary", use_container_width=True
    ):
        st.session_state["flow_step"] = 2
        st.rerun()

    st.stop()

# --- Korak 2: podešavanja + analiza + mapa ---
if flow == 2:
    st.header("Analiza i Mape")
    if choice == "have_csv":
        st.info(
            "Koristiš već pripremljene CSV fajlove — unesi putanje ispod i učitaj podatke."
        )

    with st.expander("Podešavanja analize", expanded=True):
        cmax1, cmax2, cmax3 = st.columns(3)
        with cmax1:
            max_fcd = st.number_input(
                "MAX FCD redova",
                min_value=1_000,
                max_value=5_000_000,
                value=400_000,
                step=50_000,
                key="s2_max_fcd",
            )
        with cmax2:
            max_em = st.number_input(
                "MAX emission redova",
                min_value=1_000,
                max_value=10_000_000,
                value=800_000,
                step=50_000,
                key="s2_max_em",
            )
        with cmax3:
            max_sk = st.number_input(
                "MAX vozila (skmob)",
                min_value=5,
                max_value=500,
                value=80,
                step=5,
                key="s2_max_sk",
            )
        ll1, ll2 = st.columns(2)
        with ll1:
            lat0 = st.number_input(
                "Lat0 (planar → WGS84)", value=43.3209, format="%.4f", key="s2_lat0"
            )
        with ll2:
            lon0 = st.number_input(
                "Lon0 (planar → WGS84)", value=21.8958, format="%.4f", key="s2_lon0"
            )

    st.subheader("Učitavanje CSV i analiza")
    _fcd_default = _default_fcd_csv() or st.session_state.get("fcd_path", "")
    _em_default = _default_emission_csv() or st.session_state.get("em_path", "")

    fp_fcd = st.text_input(
        "FCD CSV",
        key="inp_fcd",
        value=_fcd_default or st.session_state.get("fcd_path", ""),
    )
    fp_em = st.text_input(
        "Emission CSV",
        key="inp_em",
        value=_em_default or st.session_state.get("em_path", ""),
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Učitaj podatke", key="btn_load"):
            if not fp_fcd or not Path(fp_fcd).is_file():
                st.error("FCD CSV nije pronađen.")
            elif not fp_em or not Path(fp_em).is_file():
                st.error("Emission CSV nije pronađen.")
            else:
                try:
                    fcd_raw = read_sumo_csv(fp_fcd, nrows=int(max_fcd))
                    em_raw = apply_vehicle_name_cleanup(
                        normalize_emission_like(read_sumo_csv(fp_em, nrows=int(max_em)))
                    )
                    fcd_n, is_geo = normalize_fcd_like(fcd_raw)
                    st.session_state["fcd_path"] = fp_fcd
                    st.session_state["em_path"] = fp_em
                    st.session_state["fcd_raw"] = fcd_raw
                    st.session_state["em_raw"] = em_raw
                    st.session_state["fcd_n"] = fcd_n
                    st.session_state["is_geo"] = is_geo
                    st.session_state["merged"] = None
                    st.session_state["lengths"] = None
                    st.session_state["jump"] = None
                    st.session_state["rog"] = None
                    st.session_state["analysis_notes"] = ""
                    st.session_state["anim_prep"] = None
                    st.session_state.pop("anim_map_cached", None)
                    st.session_state.pop("anim_map_cache_key", None)
                    st.success(
                        f"Učitano: FCD {len(fcd_raw):,} redova, emission {len(em_raw):,}. "
                        f"Geo (WGS84): {'da' if is_geo else 'ne (planar x/y)'}"
                    )
                except Exception as e:
                    st.exception(e)

    with col_b:
        if st.button("Pokreni analizu (merge + MP + skmob)", key="btn_an"):
            if st.session_state["fcd_n"] is None or st.session_state["em_raw"] is None:
                st.warning("Prvo učitaj podatke.")
            else:
                try:
                    merged = merge_fcd_emission(
                        st.session_state["fcd_raw"], st.session_state["em_raw"]
                    )
                    st.session_state["merged"] = merged
                    lengths, mp_note = trajectory_lengths_movingpandas(
                        st.session_state["fcd_raw"]
                    )
                    st.session_state["lengths"] = lengths
                    jump, rog, sk_note = skmob_jump_and_rog(
                        st.session_state["fcd_raw"],
                        max_vehicles=int(max_sk),
                        lat0=float(lat0),
                        lon0=float(lon0),
                    )
                    st.session_state["jump"] = jump
                    st.session_state["rog"] = rog
                    st.session_state["analysis_notes"] = mp_note + " | " + sk_note
                    st.success("Analiza završena.")
                except Exception as e:
                    st.exception(e)

    if st.session_state.get("fcd_n") is not None:
        st.caption(
            f"Poslednji FCD: {st.session_state.get('fcd_path','')} — geo: {st.session_state.get('is_geo')}"
        )
        if st.session_state.get("analysis_notes"):
            st.info(st.session_state["analysis_notes"])

    merged = st.session_state.get("merged")
    lengths = st.session_state.get("lengths")
    jump = st.session_state.get("jump")
    rog = st.session_state.get("rog")
    map_ready = merged is not None and not merged.empty
    _folium_zoom = 15

    st.markdown("##### Rezultati analize")

    tab_tbl, tab_plot, tab_insights, tab_map, tab_mobiml, tab_time = st.tabs(
        [
            "Tabele i preuzimanja",
            "Grafici",
            "Mobilnost (RoG)",
            "Mapa (Folium)",
            "MobiML",
            "Animacija (vreme)",
        ]
    )

    with tab_tbl:
        if merged is not None and not merged.empty:
            st.subheader("Agregacija po tipu vozila")
            try:
                agg = aggregates_by_vehicle_type(merged)
                st.dataframe(agg, use_container_width=True)
                st.download_button(
                    "Preuzmi agregate (CSV)",
                    df_to_csv_bytes(agg),
                    file_name="aggregates_by_vehicle_type.csv",
                    key="dl_agg",
                )
            except Exception as e:
                st.warning(str(e))
            st.download_button(
                "Preuzmi spojeni uzorak (CSV)",
                df_to_csv_bytes(merged.head(50_000)),
                file_name="merged_sample.csv",
                key="dl_merged",
            )
        elif st.session_state.get("fcd_n") is not None:
            st.info(
                "Tabele agregacije i spojenog uzorka pojaviće se posle **Pokreni analizu**."
            )
        else:
            st.info("Prvo **Učitaj podatke**, zatim **Pokreni analizu**.")

        if lengths is not None and len(lengths):
            st.subheader("MovingPandas — dužine trajektorija (m)")
            st.dataframe(
                lengths.rename("length_m").to_frame().head(200),
                use_container_width=True,
            )
            st.download_button(
                "Preuzmi dužine (CSV)",
                df_to_csv_bytes(lengths.rename("length_m").to_frame()),
                file_name="trajectory_lengths.csv",
                key="dl_len",
            )

        if jump is not None and len(jump):
            st.subheader("scikit-mobility — jump lengths (km)")
            st.dataframe(jump.to_frame().head(500), use_container_width=True)
            st.download_button(
                "Preuzmi jump lengths (CSV)",
                df_to_csv_bytes(jump.to_frame()),
                file_name="jump_lengths.csv",
                key="dl_jump",
            )

        if rog is not None and not rog.empty:
            st.subheader("Radius of gyration (km)")
            st.dataframe(rog.head(200), use_container_width=True)
            st.download_button(
                "Preuzmi RoG (CSV)",
                df_to_csv_bytes(rog),
                file_name="radius_of_gyration.csv",
                key="dl_rog",
            )

    with tab_plot:
        if merged is None or merged.empty:
            st.info(
                "Grafici su dostupni posle učitavanja i **Pokreni analizu** (CO₂, brzina, histogrami)."
            )
        else:
            st.subheader("Emisije po tipu vozila")
            st.caption(
                "Stupci = ukupni CO₂ u učitanom uzorku. Linije = kada koji tip doprinosi (peak = najviši zbir u tom koraku). "
                "Scatter = svi redovi, boja po tipu (ne zamenjuje agregaciju)."
            )
            ft_bar = matplotlib_co2_total_by_type(merged)
            ft_lines = matplotlib_co2_timeseries_by_type(merged)
            p1, p2 = st.columns(2)
            _pyplot_in_column(p1, ft_bar)
            _pyplot_in_column(p2, ft_lines)

            st.subheader("Ostali grafici")
            fig1 = matplotlib_co2_timeseries(merged)
            fig2 = matplotlib_speed_vs_co2(merged)
            fh = (
                matplotlib_traj_length_hist(lengths)
                if lengths is not None and len(lengths)
                else None
            )
            jh = matplotlib_jump_hist(jump) if jump is not None and len(jump) else None
            row1_left, row1_right = st.columns(2)
            row2_left, row2_right = st.columns(2)
            _pyplot_in_column(row1_left, fig1)
            _pyplot_in_column(row1_right, fig2)
            _pyplot_in_column(row2_left, fh)
            _pyplot_in_column(row2_right, jh)

    with tab_insights:
        st.subheader("Mobilnost — radius pokreta, dužina i emisije")
        st.caption(
            "Spaja MovingPandas (dužina), scikit-mobility (RoG) i spojeni emission CSV po vozilu."
        )
        if not map_ready:
            st.info("Prvo **Učitaj podatke** i **Pokreni analizu**.")
        elif rog is None or rog.empty:
            st.warning(
                "Nema tabele **Radius of gyration** — povećaj **MAX vozila (skmob)** u podešavanjima "
                "i ponovo pokreni analizu."
            )
        else:
            prof = build_vehicle_profile(merged, lengths, rog)
            if prof.empty:
                st.warning("Profil vozila je prazan.")
            else:
                st.markdown(insights_narrative(prof))
                st.download_button(
                    "Preuzmi profil vozila (CSV)",
                    df_to_csv_bytes(prof),
                    file_name="vehicle_mobility_profile.csv",
                    key="dl_insights_prof",
                )
                _pyplot_in_column(st, matplotlib_rog_vs_length(prof))
                st.markdown(
                    "##### Emisije i čekanje po prostornom opsegu (RoG tertili)"
                )
                st.caption(
                    "Vozila u uzorku podeljena u **tri trećine** po radius of gyration: "
                    "*lokalno* (malo se udaljavaju od svog centra kretanja) → *široko* (pokrivaju više grada). "
                    "Broj na stubu = prosečna vrednost; `n` = koliko vozila u toj grupi."
                )
                _pyplot_wide(matplotlib_metrics_by_rog_tertile(prof))
                dirty = top_wide_and_dirty(prof, 10)
                if not dirty.empty:
                    st.subheader("Top 10 — visok RoG i visok CO₂")
                    st.dataframe(dirty, use_container_width=True)
                    st.caption(
                        "`wide_dirty_score` = normalizovan RoG × normalizovan Σ CO₂ (relativno unutar uzorka)."
                    )

    with tab_map:
        if not map_ready:
            st.caption(
                "Mapa je dostupna posle uspešnog **Pokreni analizu** (spajanje FCD + emission)."
            )

        fcd_n = st.session_state.get("fcd_n")
        if fcd_n is None:
            st.warning("Prvo učitaj FCD + emission CSV.")
        else:
            is_geo = st.session_state["is_geo"]
            merged_map = st.session_state.get("merged")
            edge_data_p = _default_edge_data()
            net_p = _default_net_xml()
            # Form: checkbox/selectbox ne pokreću rerun dok ne klikneš „Generiši mapu”.
            with st.form("map_options_form", clear_on_submit=False):
                ids = sorted(fcd_n["vehicle_id"].astype(str).unique().tolist())
                st.selectbox(
                    "Vozilo",
                    ids,
                    index=0,
                    disabled=not map_ready,
                    key="map_vid",
                    format_func=short_vehicle_id,
                )
                h_em = st.checkbox(
                    "Toplota emisija",
                    value=True,
                    key="map_heat_em",
                    disabled=not map_ready,
                )
                h_jam = st.checkbox(
                    "Toplota zastoja",
                    value=False,
                    key="map_heat_jam",
                    disabled=not map_ready,
                )
                h_speed = st.checkbox(
                    "Prosečna brzina (mreža ulica)",
                    value=False,
                    key="map_speed_net",
                    disabled=not map_ready,
                    help="Boji ulice po prosečnoj brzini iz edgeData.xml (crveno = sporo / gužva). "
                    "Koristi SUMO mrežu iz scenarija.",
                )
                gen_map = st.form_submit_button(
                    "Generiši mapu",
                    disabled=not map_ready,
                    type="primary",
                )
            if gen_map:
                if h_speed and (not edge_data_p or not net_p):
                    st.warning(
                        "Za sloj brzine potrebni su `edgeData.xml` i `osm.net.xml(.gz)` u folderu scenarija "
                        f"(`{_default_scenario_dir().name}`)."
                    )
                else:
                    try:
                        f_map = ensure_lat_lon_columns(
                            fcd_n, is_geo, float(lat0), float(lon0)
                        )
                        vid = st.session_state.get("map_vid") or (ids[0] if ids else "")
                        m = mobility_map_trajectory_and_heatmap(
                            f_map,
                            vid,
                            merged_map,
                            None,
                            is_geo=is_geo,
                            lat0=float(lat0),
                            lon0=float(lon0),
                            zoom_start=_folium_zoom,
                            show_emission_heat=h_em,
                            show_jam_heat=h_jam,
                            show_speed_network=h_speed and bool(edge_data_p and net_p),
                            edge_data_path=edge_data_p or None,
                            net_path=net_p or None,
                            rog_scale_emission=0.0,
                        )
                        if st_folium is not None:
                            st_folium(m, width=None, height=520, returned_objects=[])
                        else:
                            import streamlit.components.v1 as components

                            components.html(m._repr_html_(), height=540)
                    except Exception as e:
                        st.exception(e)

    with tab_mobiml:
        st.subheader("MobiML — trajektorije, klasifikacija, anomalije, zone")
        st.markdown("H3 zone sa visokim CO₂ i čekanjem + Folium heatmap.")
        if not map_ready:
            st.info(
                "Prvo **Učitaj podatke** i **Pokreni analizu** (merge FCD + emission)."
            )
        else:
            with st.form("mobiml_options_form", clear_on_submit=False):
                m1, m2 = st.columns(2)
                with m1:
                    ml_veh = st.number_input(
                        "MAX vozila (MobiML)",
                        min_value=20,
                        max_value=400,
                        value=120,
                        step=10,
                        key="mobiml_max_veh",
                    )
                with m2:
                    h3_res = st.slider(
                        "H3 rezolucija (zone)", 7, 11, 9, key="mobiml_h3"
                    )
                run_mobiml = st.form_submit_button(
                    "Pokreni MobiML analizu",
                    type="primary",
                )

            if run_mobiml:
                try:
                    with st.spinner("MobiML analiza (može trajati 1–3 min)…"):
                        result = run_mobiml_pipeline(
                            st.session_state["fcd_n"],
                            merged,
                            is_geo=bool(st.session_state.get("is_geo")),
                            lat0=float(lat0),
                            lon0=float(lon0),
                            max_vehicles=int(ml_veh),
                            h3_resolution=int(h3_res),
                            lengths=lengths,
                            rog=rog,
                        )
                    st.session_state["mobiml_result"] = result
                    st.success("MobiML analiza završena.")
                except Exception as e:
                    st.exception(e)

            res = st.session_state.get("mobiml_result")
            if res:
                st.caption(
                    f"Vozila: **{res.get('n_vehicles')}** · {res.get('mobiml_note', '')}"
                )
                cls = res.get("classification") or {}
                if cls.get("ok"):
                    m_acc, m_f1, m_cv = st.columns(3)
                    m_acc.metric("Tačnost (accuracy)", f"{cls['accuracy']:.1%}")
                    m_f1.metric("Macro-F1 (test)", f"{cls.get('macro_f1', 0):.3f}")
                    m_cv.metric("Macro-F1 (CV)", f"{cls.get('cv_macro_f1', 0):.3f}")
                    st.markdown(
                        f"**Model:** {cls.get('model', '?')}  \n"
                        f"**Macro-F1 (test):** {cls.get('macro_f1', 0):.3f}  \n"
                        f"**Macro-F1 (CV):** {cls.get('cv_macro_f1', 0):.3f}  \n\n"
                        "**Macro-F1 (test):** Score na test skup.  \n"
                        "**Macro-F1 (CV):** Cross-validation — model je treniran i ocenjen kroz nekoliko "
                        "nasumičnih raspodela; ovako proveravamo da li je score stabilan, ne da li smo "
                        "imali sreće kroz jednu nasumičnu raspodelu.  \n\n"
                        "Za svaki tip vozila, F1-score meri koliko model dobro nadje taj tip bez da ga "
                        "pomeša sa drugim tipovima (kombinacija precision i recall).  \n\n"
                        "Macro-F1 je aritmetička sredina tih F1 vrednosti kroz sve tipove, gde se svaki tip "
                        "računa podjednako.  \n"
                        "Za razliku od preciznosti (accuracy), model ne može da izgleda dobro tako što samo "
                        "predviđa najčešći tip (automobil)."
                    )
                    per_f1 = cls.get("per_class_f1") or {}
                    if per_f1:
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {"vehicle_type": k, "F1": f"{v:.3f}"}
                                    for k, v in sorted(per_f1.items())
                                ]
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                    cm = cls.get("confusion_matrix") or []
                    labels = cls.get("labels") or []
                    if cm and labels:
                        st.markdown(
                            "**Matrica konfuzije (test skup):**  \n"
                            "Red = stvarni tip  \n"
                            "Kolona = predviđeni tip"
                        )
                        st.dataframe(
                            pd.DataFrame(cm, index=labels, columns=labels),
                            use_container_width=True,
                        )
                elif cls.get("message"):
                    st.warning(cls["message"])

                hot = res.get("hotspots") or []
                if hot:
                    st.subheader("Top zone (H3) — visok CO₂ / čekanje")
                    st.caption(
                        "Tamnija boja = veće emisije, veća koncentracija saobraćaja. "
                        "Svaki šestougaonik je jedna H3 ćelija. Žuto → crveno = viši planner_score. "
                        "Prikazuje najkoncentrovanije zone emisije i zakrčenja saobraćaja."
                    )
                    try:
                        m_h = folium_h3_hotspots_map(pd.DataFrame(hot), zoom_start=14)
                        if st_folium is not None:
                            st_folium(m_h, width=None, height=420, returned_objects=[])
                        else:
                            import streamlit.components.v1 as components

                            components.html(m_h._repr_html_(), height=440)
                    except Exception as ex:
                        st.warning(f"Mapa zona: {ex}")
                    st.dataframe(pd.DataFrame(hot), use_container_width=True)

                an = res.get("anomalies") or {}
                if an.get("ok") and an.get("anomalies"):
                    st.subheader(f"Anomalna vozila ({an.get('n_anomalies', 0)})")
                    st.caption(
                        "Vozila čiji rezime puta i emisija odstupaju od ostatka uzorka (automatska detekcija)."
                    )
                    st.dataframe(
                        pd.DataFrame(an["anomalies"]), use_container_width=True
                    )

                out_json = ROOT / "sumo_output" / "mobiml_results.json"
                if out_json.is_file():
                    st.download_button(
                        "Preuzmi mobiml_results.json",
                        out_json.read_bytes(),
                        file_name="mobiml_results.json",
                        key="dl_mobiml_json",
                    )

    with tab_time:
        st.subheader("Kretanje kroz vreme — vozila i gužve na ulicama")
        st.markdown(
            "Prikazuje timestepove simulacije i promenu prohodnosti ulica kroz vreme."
        )
        fcd_raw_anim = st.session_state.get("fcd_raw")
        if fcd_raw_anim is None:
            st.info("Prvo **Učitaj podatke** (FCD CSV).")
        else:
            net_anim = _default_net_xml()
            with st.form("anim_prep_form", clear_on_submit=False):
                a1, a2, a3 = st.columns(3)
                with a1:
                    anim_step = st.slider(
                        "Korak vremena (s)",
                        min_value=1,
                        max_value=60,
                        value=10,
                        key="anim_step_sec",
                    )
                with a2:
                    anim_frames = st.slider(
                        "Maks. broj kadrova",
                        min_value=12,
                        max_value=480,
                        value=120,
                        step=12,
                        key="anim_max_frames",
                    )
                with a3:
                    anim_max_veh = st.number_input(
                        "Maks. vozila po kadru",
                        min_value=50,
                        max_value=1200,
                        value=450,
                        step=50,
                        key="anim_max_veh",
                    )
                anim_show_veh = st.checkbox(
                    "Prikaži vozila", value=True, key="anim_show_veh"
                )
                prep_anim = st.form_submit_button("Pripremi animaciju", type="primary")

            if prep_anim:
                if not net_anim:
                    st.warning(
                        "SUMO mreža (`osm.net.xml.gz`) nije u scenariju — priprema bez boja ulica (samo vozila)."
                    )
                try:
                    with st.spinner("Priprema kadrova (agregacija po vremenu)…"):
                        prepared = prepare_time_animation(
                            fcd_raw_anim,
                            is_geo=bool(st.session_state.get("is_geo")),
                            lat0=float(lat0),
                            lon0=float(lon0),
                            net_path=net_anim or None,
                            step_sec=float(anim_step),
                            max_frames=int(anim_frames),
                            max_vehicles_per_frame=int(anim_max_veh),
                            show_vehicles=anim_show_veh,
                            show_street_speeds=bool(net_anim),
                        )
                    st.session_state["anim_prep"] = prepared
                    st.session_state.pop("anim_map_cached", None)
                    st.session_state.pop("anim_map_cache_key", None)
                    meta = prepared["meta"]
                    st.success(
                        f"Spremno: {meta['n_frames']} kadrova "
                        f"({meta['time_min']:.0f}s–{meta['time_max']:.0f}s), "
                        f"~{prepared_json_size_kb(prepared):.0f} KB GeoJSON."
                    )
                except Exception as e:
                    st.exception(e)

            prepared_anim = st.session_state.get("anim_prep")
            if (
                prepared_anim
                and prepared_anim.get("meta", {}).get("geo_version", 0) < 6
            ):
                prepared_anim = None
                st.session_state["anim_prep"] = None
                st.session_state.pop("anim_map_cached", None)
                st.session_state.pop("anim_map_cache_key", None)
                st.warning(
                    "Ponovo klikni **Pripremi animaciju** (ažuriran prikaz vozila)."
                )
            if prepared_anim:
                meta = prepared_anim["meta"]
                st.caption(
                    f"Kadrovi: **{meta['n_frames']}** · korak ~**{meta['step_sec']:.0f} s** · "
                    f"vozila: {meta['n_vehicle_features']:,} tačaka · "
                    f"ulice: {meta['n_edge_features']:,} segmenata · "
                    f"crveno = sporo (&lt; ~{meta['jam_speed_kmh']:.0f} km/h)."
                )
                has_street_data = bool(
                    (prepared_anim.get("edge_geojson") or {}).get("features")
                )
                anim_display_streets = st.checkbox(
                    "Prikaži boje ulica (gužva)",
                    value=True,
                    key="anim_display_streets",
                    disabled=not has_street_data,
                )
                if not has_street_data:
                    st.caption(
                        "Boje ulica nisu u pripremljenim podacima (nedostaje mreža ili **Pripremi** ponovo)."
                    )
                try:
                    # Keš mape — ne graditi 3MB HTML na svaki rerun cele aplikacije
                    anim_map_key = (
                        prepared_anim.get("meta", {}).get("geo_version"),
                        id(prepared_anim),
                        bool(anim_display_streets),
                    )
                    if st.session_state.get("anim_map_cache_key") != anim_map_key:
                        st.session_state["anim_map_cached"] = folium_time_animation_map(
                            prepared_anim,
                            zoom_start=14,
                            show_street_speeds=anim_display_streets,
                        )
                        st.session_state["anim_map_cache_key"] = anim_map_key
                    m_anim = st.session_state["anim_map_cached"]
                    if st_folium is not None:
                        st_folium(m_anim, width=None, height=560, returned_objects=[])
                    else:
                        import streamlit.components.v1 as components

                        components.html(m_anim._repr_html_(), height=580)
                except Exception as e:
                    st.exception(e)

elif flow == 1:
    st.warning("Nedosledan tok — koristi **Početak** u bočnoj traci.")
else:
    st.error("Nepoznato stanje toka — koristi **Početak** u bočnoj traci.")
