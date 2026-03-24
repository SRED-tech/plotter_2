import re
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Helpers to parse CSV (wide or tidy) ----------

def _coerce_time(col):
    """Coerce a 'time' column to float hours."""
    c = pd.to_numeric(col, errors="coerce")
    if c.notna().all():
        return c.astype(float)

    # If Incucyte-like strings, try to pull out numbers
    s = col.astype(str).str.extract(r"([0-9]*\.?[0-9]+)")[0]
    return pd.to_numeric(s, errors="coerce").astype(float)


def _base_group_name(colname: str) -> str:
    """Strip off common replicate suffixes, e.g. A_R1 -> A."""
    m = re.match(r"^(.*)_(R\d+|Rep\d+|rep\d+)$", str(colname))
    return m.group(1) if m else str(colname)


def _read_text_buffer(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin1")


def _detect_incucyte_export(raw_text: str) -> bool:
    lines = raw_text.splitlines()
    if len(lines) < 2:
        return False

    first = lines[0].strip()
    second = lines[1].strip()

    return (
        first.startswith("Vessel Name:")
        and "Date Time" in second
        and "Elapsed" in second
    )


def _clean_incucyte_group_name(colname: str) -> str:
    """
    Example:
    'e6y3_purifiedNS (1) 35K / well Unstimulated 100 mg/ml (D1)'
    -> 'e6y3_purifiedNS - Unstimulated'
    """
    s = str(colname)

    # remove trailing well ID like (D1)
    s = re.sub(r"\s*\([A-Z]\d+\)\s*$", "", s)

    # split sample name from condition
    s = re.sub(r"\s*\(\d+\)\s*\d+\s*K\s*/\s*well\s*", " | ", s, flags=re.IGNORECASE)

    # remove concentration text
    s = re.sub(r"\s+\d+\s*mg/ml", "", s, flags=re.IGNORECASE)

    s = s.replace(" | ", " - ").strip()
    return s


def parse_incucyte_export(uploaded_file) -> pd.DataFrame:
    """
    Parse Incucyte TXT export:
    - first row metadata ('Vessel Name: ...')
    - second row header
    - tab-delimited data
    """
    text = _read_text_buffer(uploaded_file)
    df = pd.read_csv(io.StringIO(text), sep="\t", skiprows=1)

    if "Elapsed" not in df.columns:
        raise ValueError("Incucyte export detected, but 'Elapsed' column was not found.")

    value_cols = [c for c in df.columns if c not in ["Date Time", "Elapsed"]]

    long = df.melt(
        id_vars=["Elapsed"],
        value_vars=value_cols,
        var_name="raw_col",
        value_name="value",
    ).rename(columns={"Elapsed": "time"})

    # clean group names
    long["group"] = long["raw_col"].apply(_clean_incucyte_group_name)

    # assign replicates by column order within each group
    col_map = pd.DataFrame({"raw_col": value_cols})
    col_map["group"] = col_map["raw_col"].apply(_clean_incucyte_group_name)
    col_map["replicate_num"] = col_map.groupby("group").cumcount() + 1
    col_map["replicate"] = "R" + col_map["replicate_num"].astype(str)

    long = long.merge(
        col_map[["raw_col", "replicate"]],
        on="raw_col",
        how="left",
        validate="many_to_one",
    )

    # preserve group order from file
    group_order = pd.unique(col_map["group"]).tolist()
    long["group"] = pd.Categorical(long["group"], categories=group_order, ordered=True)

    long["time"] = _coerce_time(long["time"])
    long["replicate"] = long["replicate"].astype(str)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    long = long.dropna(subset=["time", "value"])
    return long[["time", "group", "replicate", "value"]].reset_index(drop=True)


def read_incucyte_csv(path_or_buffer) -> pd.DataFrame:
    """
    Read either:
      A) WIDE format: 'time' + one column per group (e.g., Time,A,B,C)
         Optional replicates via suffixes: A_R1, A_R2, B_R1 ...
      B) TIDY format: time, group, replicate, value
      C) Incucyte TXT export: metadata row + tab-delimited well columns

    Returns a tidy DataFrame with columns: time, group, replicate, value
    """
    # --- detect Incucyte TXT export first ---
    if hasattr(path_or_buffer, "getvalue"):
        raw_text = _read_text_buffer(path_or_buffer)
        if _detect_incucyte_export(raw_text):
            return parse_incucyte_export(path_or_buffer)

    df = pd.read_csv(path_or_buffer)
    lower_map = {c.lower(): c for c in df.columns}
    cols_lower = set(lower_map.keys())

    # --------- WIDE format ---------
    if "time" in cols_lower and not {"group", "replicate", "value"}.issubset(cols_lower):
        time_col = lower_map["time"]
        value_cols_in_order = [c for c in df.columns if c != time_col]
        df = df.rename(columns={time_col: "time"})

        long = df.melt(
            id_vars="time",
            value_vars=value_cols_in_order,
            var_name="col",
            value_name="value",
        )

        # detect replicate suffixes
        m = long["col"].astype(str).str.extract(r"^(.*)_(R\d+|Rep\d+|rep\d+)$")
        has_rep = m.notna().all(axis=1)

        long["group"] = np.where(has_rep, m[0], long["col"].astype(str))
        long["replicate"] = np.where(has_rep, m[1], "R1")

        # preserve base group order from header
        group_order = []
        seen = set()
        for c in value_cols_in_order:
            base = _base_group_name(c)
            if base not in seen:
                group_order.append(base)
                seen.add(base)

        long["group"] = long["group"].astype(str)
        long["group"] = pd.Categorical(long["group"], categories=group_order, ordered=True)

        long["time"] = _coerce_time(long["time"])
        long["replicate"] = long["replicate"].astype(str)
        long["value"] = pd.to_numeric(long["value"], errors="coerce")

        long = long.dropna(subset=["time", "value"])
        return long[["time", "group", "replicate", "value"]].reset_index(drop=True)

    # --------- TIDY format ---------
    required = {"time", "group", "value"}
    if required.issubset(cols_lower):
        # normalise names
        df = df.rename(columns={
            lower_map["time"]: "time",
            lower_map["group"]: "group",
            lower_map["value"]: "value",
        })
        if "replicate" in cols_lower:
            df = df.rename(columns={lower_map["replicate"]: "replicate"})
        else:
            df["replicate"] = "R1"

        df["time"] = _coerce_time(df["time"])
        df["group"] = df["group"].astype(str)
        df["replicate"] = df["replicate"].astype(str)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df = df.dropna(subset=["time", "value"])
        return df[["time", "group", "replicate", "value"]].reset_index(drop=True)

    raise ValueError(
        "Could not detect wide or tidy format. "
        "Need either:\n"
        "  Wide: 'time' + one column per group\n"
        "  Tidy: time, group, (replicate), value\n"
        "  Or Incucyte TXT export with Vessel Name / Date Time / Elapsed"
    )


def aggregate_stats(df: pd.DataFrame, interval_hours: float = None) -> pd.DataFrame:
    """
    Aggregate replicates to group-level mean ± SD/SEM.
    If interval_hours is not None, bin time into that spacing (e.g., 4 h).
    """
    d = df.copy()
    if interval_hours is not None and interval_hours > 0:
        d["time_bin"] = (
            np.floor(d["time"] / interval_hours) * interval_hours
        ).astype(float)
        time_col = "time_bin"
    else:
        time_col = "time"

    group_stats = (
        d.groupby(["group", time_col], as_index=False)["value"]
        .agg(mean="mean", sd="std", n="count")
    )
    group_stats = group_stats.rename(columns={time_col: "time"})
    group_stats["sd"] = group_stats["sd"].fillna(0.0)
    group_stats["sem"] = group_stats["sd"] / np.sqrt(group_stats["n"].clip(lower=1))
    return group_stats


def make_color_list(n: int):
    """
    Return at least n colors.
    Uses matplotlib categorical palettes and expands beyond 10 groups safely.
    """
    cmap_names = ["tab20", "tab20b", "tab20c"]
    colors = []

    for cmap_name in cmap_names:
        cmap = plt.get_cmap(cmap_name)
        for i in range(cmap.N):
            c = cmap(i)
            hex_color = "#{0:02x}{1:02x}{2:02x}".format(
                int(c[0] * 255),
                int(c[1] * 255),
                int(c[2] * 255),
            )
            colors.append(hex_color)

    if n <= len(colors):
        return colors[:n]

    repeats = (n + len(colors) - 1) // len(colors)
    return (colors * repeats)[:n]


def save_figure_bytes(fig, fmt="png", dpi=300):
    """Save matplotlib figure to bytes for download."""
    buf = io.BytesIO()
    save_kwargs = {"format": fmt, "bbox_inches": "tight"}

    if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
        save_kwargs["dpi"] = dpi

    fig.savefig(buf, **save_kwargs)
    buf.seek(0)
    return buf


def get_download_mime(fmt: str) -> str:
    fmt = fmt.lower()
    mime_map = {
        "png": "image/png",
        "pdf": "application/pdf",
        "svg": "image/svg+xml",
        "tif": "image/tiff",
        "tiff": "image/tiff",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }
    return mime_map.get(fmt, "application/octet-stream")


# ---------- Streamlit UI ----------

st.title("Incucyte Timecourse Plotter")

st.markdown(
    """
You can either **upload a CSV** *or* **enter the data manually**.

**Manual/tidy format columns:**

- `time` – time in hours (0, 2, 4, 6, …)  
- `group` – condition / treatment name (e.g. Control, DrugA)  
- `replicate` – replicate ID (e.g. R1, R2, R3)  
- `value` – confluence / intensity / whatever you measured
"""
)

input_mode = st.radio(
    "How do you want to provide data?",
    ["Upload CSV", "Enter data manually"],
    index=0,
)

tidy = None

# --------- MODE 1: Upload CSV ---------
if input_mode == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV or Incucyte TXT", type=["csv", "txt", "tsv"])

    if uploaded is not None:
        try:
            tidy = read_incucyte_csv(uploaded)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.info("Upload a CSV to begin.")

# --------- MODE 2: Manual entry ---------
else:
    st.markdown("### Enter your data")
    st.write(
        "You can **type directly** or **paste from Excel**. "
        "Add/remove rows as needed."
    )

    starter = pd.DataFrame(
        {
            "time": [0.0, 0.0, 2.0, 2.0],
            "group": ["Control", "DrugA", "Control", "DrugA"],
            "replicate": ["R1", "R1", "R1", "R1"],
            "value": [np.nan, np.nan, np.nan, np.nan],
        }
    )

    edited = st.data_editor(
        starter,
        num_rows="dynamic",
        use_container_width=True,
        key="manual_editor",
    )

    edited["time"] = pd.to_numeric(edited["time"], errors="coerce")
    edited["value"] = pd.to_numeric(edited["value"], errors="coerce")
    edited["group"] = edited["group"].astype(str)
    edited["replicate"] = edited["replicate"].astype(str)

    tidy = edited.dropna(subset=["time", "value"]).reset_index(drop=True)

    if tidy.empty:
        st.warning("Fill in at least some rows (time + value) to generate plots.")

# ---------- If we have tidy data, proceed ----------

if tidy is not None and not tidy.empty:
    st.subheader("Parsed data (tidy format)")
    st.dataframe(tidy.head())

    groups = pd.unique(tidy["group"].astype(str)).tolist()
    default_colors = make_color_list(len(groups))

    # ---------- Sidebar settings ----------
    st.sidebar.header("Plot settings")

    x_label = st.sidebar.text_input("X axis label", value="Time (h)")
    y_label = st.sidebar.text_input("Y axis label", value="Confluence / Intensity")

    interval = st.sidebar.number_input(
        "Time binning (hours, 0 = no binning)",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )
    interval_hours = interval if interval > 0 else None

    error_choice = st.sidebar.selectbox(
        "Error band",
        ["SEM", "SD", "None"],
        index=0,
    )

    smooth_step = st.sidebar.number_input(
        "Plot every Nth timepoint (visual smoothing)",
        min_value=1,
        value=1,
        step=1,
        help="1 = use all points; 2 = every 2nd; 3 = every 3rd; etc.",
    )

    show_replicates_on_mean = st.sidebar.checkbox(
        "Overlay faint replicate lines on mean plot",
        value=False,
    )

    replicate_alpha = st.sidebar.slider(
        "Replicate line opacity",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )

    band_alpha = st.sidebar.slider(
        "Error band opacity",
        min_value=0.05,
        max_value=0.8,
        value=0.20,
        step=0.05,
    )

    line_width = st.sidebar.slider(
        "Mean line width",
        min_value=1.0,
        max_value=5.0,
        value=2.5,
        step=0.5,
    )

    st.sidebar.markdown("### Figure size")
    figure_preset = st.sidebar.selectbox(
        "Export preset",
        ["Custom", "Screen", "Presentation", "Publication single-column", "Publication double-column"],
        index=3,
    )

    preset_dims = {
        "Custom": (8.0, 5.0),
        "Screen": (8.0, 5.0),
        "Presentation": (10.0, 6.0),
        "Publication single-column": (3.35, 2.6),
        "Publication double-column": (6.9, 4.8),
    }

    default_w, default_h = preset_dims[figure_preset]

    fig_width = st.sidebar.number_input(
        "Figure width (inches)",
        min_value=2.0,
        max_value=20.0,
        value=float(default_w),
        step=0.1,
    )
    fig_height = st.sidebar.number_input(
        "Figure height (inches)",
        min_value=2.0,
        max_value=20.0,
        value=float(default_h),
        step=0.1,
    )

    st.sidebar.markdown("### Publication export")
    export_format = st.sidebar.selectbox(
        "Download format",
        ["PNG", "PDF", "SVG", "TIFF"],
        index=0,
    )
    export_dpi = st.sidebar.selectbox(
        "Raster DPI",
        [150, 300, 600, 1200],
        index=2,
        help="Used for PNG/TIFF only. PDF/SVG are vector exports.",
    )

    st.sidebar.markdown("### Groups")
    st.sidebar.caption("Rename groups and set colours manually.")

    name_map = {}
    color_map = {}
    for i, g in enumerate(groups):
        with st.sidebar.expander(f"{g}", expanded=False):
            display_name = st.text_input(
                f"Display name for {g}",
                value=g,
                key=f"display_name_{i}",
            )
            chosen_color = st.color_picker(
                f"Colour for {g}",
                value=default_colors[i],
                key=f"color_{i}",
            )
        name_map[g] = display_name
        color_map[g] = chosen_color

    group_df = pd.DataFrame(
        {
            "group": groups,
            "display_name": [name_map[g] for g in groups],
            "color": [color_map[g] for g in groups],
        }
    )

    # ---------- Compute stats ----------
    stats = aggregate_stats(tidy, interval_hours=interval_hours)

    stats = stats.merge(
        group_df, on="group", how="left", validate="many_to_one"
    )
    tidy_merged = tidy.merge(
        group_df, on="group", how="left", validate="many_to_one"
    )

    error_col = None
    plot_title_suffix = ""
    if error_choice == "SD":
        error_col = "sd"
        plot_title_suffix = " ± SD"
    elif error_choice == "SEM":
        error_col = "sem"
        plot_title_suffix = " ± SEM"

    # ---------- Plot: mean ± chosen error ----------
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if show_replicates_on_mean:
        for (g, r), sub in tidy_merged.groupby(["group", "replicate"], sort=False):
            sub = sub.sort_values("time")
            color = sub["color"].iloc[0] if pd.notna(sub["color"].iloc[0]) else None
            ax.plot(
                sub["time"],
                sub["value"],
                color=color,
                alpha=replicate_alpha,
                linewidth=1.0,
                zorder=1,
            )

    for g, sub in stats.groupby("group", sort=False):
        sub = sub.sort_values("time")
        name = sub["display_name"].iloc[0]
        color = sub["color"].iloc[0] if pd.notna(sub["color"].iloc[0]) else None

        if smooth_step > 1:
            sub_plot = sub.iloc[::smooth_step].copy()
        else:
            sub_plot = sub

        ax.plot(
            sub_plot["time"],
            sub_plot["mean"],
            label=name,
            color=color,
            linewidth=line_width,
            zorder=3,
        )

        if error_col is not None and sub_plot[error_col].notna().any():
            ax.fill_between(
                sub_plot["time"],
                sub_plot["mean"] - sub_plot[error_col],
                sub_plot["mean"] + sub_plot[error_col],
                alpha=band_alpha,
                color=color,
                zorder=2,
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    st.subheader(f"Mean{plot_title_suffix} per group")
    st.pyplot(fig)

    fmt = export_format.lower()
    if fmt == "tiff":
        fmt = "tif"

    mean_buf = save_figure_bytes(fig, fmt=fmt, dpi=export_dpi)
    st.download_button(
        f"Download mean plot ({export_format}, {'vector' if export_format in ['PDF', 'SVG'] else f'{export_dpi} dpi'})",
        data=mean_buf,
        file_name=f"incucyte_mean_{error_choice.lower()}.{fmt}",
        mime=get_download_mime(fmt),
    )

    # Optional quick-access publication downloads
    png300_buf = save_figure_bytes(fig, fmt="png", dpi=300)
    png600_buf = save_figure_bytes(fig, fmt="png", dpi=600)
    pdf_buf = save_figure_bytes(fig, fmt="pdf", dpi=export_dpi)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Quick download PNG 300 dpi",
            data=png300_buf,
            file_name=f"incucyte_mean_{error_choice.lower()}_300dpi.png",
            mime="image/png",
        )
    with col2:
        st.download_button(
            "Quick download PNG 600 dpi",
            data=png600_buf,
            file_name=f"incucyte_mean_{error_choice.lower()}_600dpi.png",
            mime="image/png",
        )
    with col3:
        st.download_button(
            "Quick download PDF (vector)",
            data=pdf_buf,
            file_name=f"incucyte_mean_{error_choice.lower()}.pdf",
            mime="application/pdf",
        )

    # ---------- Plot: replicate spaghetti ----------
    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))

    for (g, r), sub in tidy_merged.groupby(["group", "replicate"], sort=False):
        sub = sub.sort_values("time")
        name = sub["display_name"].iloc[0]
        color = sub["color"].iloc[0] if pd.notna(sub["color"].iloc[0]) else None
        ax2.plot(
            sub["time"],
            sub["value"],
            color=color,
            alpha=replicate_alpha,
            linewidth=1.5,
            label=name,
        )

    # deduplicate legend labels
    handles, labels = ax2.get_legend_handles_labels()
    seen = set()
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)

    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.legend(
        new_handles,
        new_labels,
        title="Group",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    ax2.grid(True, alpha=0.3)

    st.subheader("Replicate spaghetti plot")
    st.pyplot(fig2)

    spag_buf = save_figure_bytes(fig2, fmt=fmt, dpi=export_dpi)
    st.download_button(
        f"Download spaghetti plot ({export_format}, {'vector' if export_format in ['PDF', 'SVG'] else f'{export_dpi} dpi'})",
        data=spag_buf,
        file_name=f"incucyte_spaghetti.{fmt}",
        mime=get_download_mime(fmt),
    )

    # ---------- Summary table + CSV ----------
    st.subheader("Summary table")
    st.dataframe(
        stats[["group", "display_name", "time", "mean", "sd", "sem", "n"]]
        .sort_values(["group", "time"])
        .reset_index(drop=True)
    )

    csv_buffer = io.StringIO()
    stats.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download summary CSV",
        data=csv_buffer.getvalue(),
        file_name="incucyte_summary_mean_sd_sem.csv",
        mime="text/csv",
    )
