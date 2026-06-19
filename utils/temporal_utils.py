import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

FILES = {
    "hb":   "HB.xlsx",
    "kr":   "KR.xlsx",
    "mlpt": "MLPT.xlsx",
    "gt":   "GT.xlsx",
    "svb":  "SVB.xlsx",
    "runt": "run.xlsx",
    "odd":  "odd.xlsx",
}

COLORS = {
    "hb":    "blue",
    "kr":    "red",
    "mlpt":  "green",
    "gt":    "gold",
    "svb":   "brown",
    "runt":  "hotpink",
    "odd":   "cyan",
    "cad":   "violet",
    "en":    "olive",
    "eve":   "purple",
}

def lighten_color(color, amount=0.5):
    c = np.array(to_rgb(color))
    white = np.ones(3)
    return tuple((1 - amount) * c + amount * white)

def scale_to_unit(data_dict):
    stage_means = []
    for v in data_dict.values():
        vals = np.array(v, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            stage_means.append(vals.mean())
    if len(stage_means) == 0:
        return data_dict
    mn, mx = np.min(stage_means), np.max(stage_means)
    if mx - mn == 0:
        return data_dict
    scaled = {}
    for k, v in data_dict.items():
        vals = np.array(v, dtype=float)
        scaled[k] = (vals - mn) / (mx - mn)
    return scaled

def compute_mean_se(data_dict, x_stages):
    xs, ys, ses = [], [], []
    for i, s in enumerate(x_stages):
        vals = np.array(data_dict.get(s, []), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        xs.append(i)
        ys.append(vals.mean())
        ses.append(vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0)
    return np.array(xs), np.array(ys), np.array(ses)

def smooth_plot(ax, xs, ys, ses, color, label=None, show_variance=True):
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys, ses = xs[mask], ys[mask], ses[mask]
    if len(xs) < 3:
        return
    sx = np.linspace(xs.min(), xs.max(), 400)
    spline = PchipInterpolator(xs, ys)
    sy = spline(sx)
    ax.plot(sx, sy, color=color, lw=2, label=label)
    if show_variance and len(ses) > 0:
        se_spline = PchipInterpolator(xs, ses)
        se_vals = se_spline(sx)
        ax.fill_between(sx, sy - se_vals, sy + se_vals, color=color, alpha=0.2)

def read_one(filepath):
    df = pd.read_excel(filepath, sheet_name="BASE")
    df = df.replace(0.0, np.nan)
    df["Stage"] = df["Stage"].astype(str).str.strip()
    stages = df["Stage"].tolist()
    eve_cols = [c for c in df.columns if c.lower().startswith("eve")]
    mrna_cols = [c for c in df.columns if c.lower().startswith("mrna")]
    full_stages = [s for s in stages]  # Remove phi
    eve_d = {s: [] for s in full_stages}
    mrna_d = {s: [] for s in full_stages}
    for s in stages:
        row = df[df["Stage"] == s]
        if not row.empty:
            eve_vals = row[eve_cols].values.flatten()
            mrna_vals = row[mrna_cols].values.flatten()
            eve_d[s] = [float(v) for v in eve_vals if pd.notna(v)]
            mrna_d[s] = [float(v) for v in mrna_vals if pd.notna(v)]
    return full_stages, eve_d, mrna_d

def load_all_data(data_path="data/Temporal"):
    global_stage_order = []
    per_gene = {}
    for gene, fname in FILES.items():
        stages, eve_d, mrna_d = read_one(f"{data_path}/{fname}")
        per_gene[gene] = {"eve": eve_d, "mrna": mrna_d}
        for s in stages:
            if s not in global_stage_order:
                global_stage_order.append(s)
    pooled_eve = {s: [] for s in global_stage_order}
    for gene in per_gene:
        for s in global_stage_order:
            pooled_eve[s].extend(per_gene[gene]["eve"].get(s, []))
    pooled_eve_scaled = scale_to_unit(pooled_eve)
    scaled_genes = {}
    for gene in per_gene:
        scaled_genes[gene] = scale_to_unit(per_gene[gene]["mrna"])
    return global_stage_order, pooled_eve_scaled, scaled_genes