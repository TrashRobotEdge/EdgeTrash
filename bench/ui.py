"""Benchmark UI: pick the config, install/check it, run, look at the results.
A thin layer over the same scripts the command line uses: nothing here is UI-only.

    .venv/bin/streamlit run bench/ui.py --server.address 0.0.0.0   # setup/setup.sh starts it for you
"""

import subprocess
import sys

import pandas as pd
import streamlit as st

from edgetrash.detector import MODEL_DIR, read_manifest
from meta import REPO, run_meta
from run import load_config
from sweep import available

CONFIGS = REPO / "bench" / "configs"
PY = sys.executable


def stream(cmd: list[str]) -> bool:
    """Run a command from the repo root, show its output live. True if it succeeded."""
    st.caption("$ " + " ".join(str(c) for c in cmd))
    box, lines = st.empty(), []
    with subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        for line in p.stdout:
            lines.append(line.rstrip())
            box.code("\n".join(lines[-40:]))
    (st.success if p.returncode == 0 else st.error)(f"exit code {p.returncode}")
    return p.returncode == 0


st.set_page_config(page_title="EdgeTrash benchmark", layout="wide")
st.title("EdgeTrash benchmark")
meta = run_meta()
st.caption(
    f"{meta['model'] or meta['host']} · {meta['os']} · governor: {meta['governor']} · commit {meta['commit'][:7]}"
    + (" (uncommitted changes!)" if meta["dirty"] else "")
)

st.header("1. Config")
names = sorted(p.stem for p in CONFIGS.glob("*.yaml"))
name = st.selectbox("Which config is this machine?", names)
cfg_path = CONFIGS / f"{name}.yaml"
cfg = load_config(cfg_path)
if cfg.get("label"):
    st.info(cfg["label"])
if st.button("Install / check this config (runs setup/setup.sh)"):
    stream(["bash", "setup/setup.sh", "--config", name, "--no-ui", "--yes"])

st.header("2. Models on this machine")
manifest = read_manifest()
ok, missing = available(cfg["variants"])
st.dataframe(
    pd.DataFrame(
        [
            {
                "variant": v,
                "status": "ready" if v in ok else ("not in manifest" if v not in manifest else "file missing: copy it into models/files/"),
                "file": manifest.get(v, {}).get("file", "-"),
                "MB": manifest.get(v, {}).get("size_mb"),
            }
            for v in cfg["variants"]
        ]
    ),
    hide_index=True,
    width="stretch",
)
st.caption(f"Model folder: {MODEL_DIR}. Bring files with the USB bundle (setup/README.md).")

st.header("3. Run")
chosen = st.multiselect("Variants", ok, default=ok)
reps = st.number_input("Repetitions", 1, 20, int(cfg.get("reps", 5)))
c1, c2, c3 = st.columns(3)
if c1.button("Smoke test (seconds)"):
    stream([PY, "bench/smoke.py", "--config", str(cfg_path)])
if c2.button(f"Idle baseline ({cfg.get('idle_s', 60)} s, don't touch the device)"):
    stream([PY, "bench/run.py", "--config", str(cfg_path), "--idle"])
if c3.button(f"Run benchmark ({len(chosen)} variants x {reps} reps, ~{len(chosen) * reps * (cfg.get('duration_s', 60) + 20) // 60} min)"):
    import random

    for rep in range(reps):
        order = chosen[:]
        random.Random(rep).shuffle(order)  # interleaved, same order as bench/sweep.py
        for v in order:
            if not stream([PY, "bench/run.py", "--config", str(cfg_path), "--variant", v, "--rep", str(rep)]):
                st.stop()

st.header("4. Results")
if st.button("Analyze (regenerates tables + figures from results/raw/)"):
    stream([PY, "bench/analyze.py"])
summary = REPO / "results" / "summary.csv"
if summary.exists():
    s = pd.read_csv(summary)
    s = s[s.device == cfg["device"]]
    wanted = ["variant", "reps", "p50_ms", "p95_ms", "p95_iqr_ms", "p99_ms", "fps", "cpu_pct", "temp_max_c", "pmic_w", "pmic_mj_per_inf"]
    cols = [c for c in wanted if c in s]
    st.dataframe(s[cols].round(2), hide_index=True, width="stretch")
    for fig in sorted((REPO / "results" / "figures").glob(f"*_{cfg['device']}*.png")):
        st.image(str(fig), caption=fig.name)
