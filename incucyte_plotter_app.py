import re
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Helpers to parse CSV / TSV / Incucyte export ----------

def _coerce_time(col):
    """Coerce a 'time' column to float hours."""
    c = pd.to_numeric(col, errors="coerce")
    if c.notna().all():
        return c.astype(float)

    s = col.astype(str).str.extract(r"([0-9]*\.?[0-9]+)")[0]
    return pd.to_numeric(s, errors="coerce").astype(float)


def _base_group_name(colname: str) -> str:
    """Strip off common replicate suffixes, e.g. A_R1 -> A."""
    m = re.match(r"^(.*)_(R\d+|Rep\d+|rep\d+)$", str(colname))
    return m.group(1) if m else str(colname)


def _clean_incucyte_group_name(colname: str) -> str:
    """
    Convert an Incucyte export column name like:
    'e6y3_purifiedNS (1) 35K / well Unstimulated 100 mg/ml (D1)'
    into a cleaner group name like:
    'e6y3_purifiedNS - Unstimulated'
    """
    s = str(colname)

    # Remove trailing well ID like (D1)
    s = re.sub(r"\s*\([A-Z]\d+\)\s*$", "", s)

    # Remove common density / well text
    s = re.sub(r"\s*\(\d+\)\s*\d+\s*K\s*/\s*well\s*", " | ", s, flags=re.IGNORECASE)

    # Remove dosage-like phrases
    s = re.sub(r"\s+\d+\s*mg/ml", "", s, flags=re.IGNORECASE)

    # Normalize separators
    s = s.replace(" | ", " - ").strip()

    return s


def _extract_well_id(colname: str) -> str:
    """Extract well ID from an Incucyte export column name, e.g. (D1) -> D1."""
    m = re.search(r"\(([A-Z]\d+)\)\s*$", str(colname))
    return m.group(1) if m else ""


def _detect_incucyte_export(raw_text: str) -> bool:
    """
    Detect Incucyte text export by metadata + header pattern.
    """
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


def _read_text_buffer(uploaded_file) -> str:
    """Read uploaded file safely as text and restore pointer."""
    raw = uploaded_file.getvalue()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin1")


def parse_incucyte_export(path_or_buffer) -> pd.DataFrame:
    """
    Parse Incucyte text export like:
    row 1: metadata (e.g. Vessel Name: ...)
    row 2: tab-delimited header with Date Time, Elapsed, and well columns

    Returns tidy dataframe with columns: time, group, replicate, value
    """
    if hasattr(path_or_buffer, "getvalue"):
        text = _read_text_buffer(path_or_buffer)
        data_io = io.StringIO(text)
    else:
        with open(path_or_buffer, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        data_io = io.StringIO(text)

    df = pd.read_csv(data_io, sep="\t", skiprows=1)

    if "Elapsed" not in df.columns:
        raise ValueError("Incucyte export detected, but could not find 'Elapsed' column.")

    # Keep only data columns
    value_cols = [c for c in df.columns if c not in ["Date Time", "Elapsed"]]

    long = df.melt(
        id_vars=["Elapsed"],
        value_vars=value_cols,
        var_name="raw_col",
        value_name="value",
    )

    long = long.rename(columns={"Elapsed": "time"})

    # Group name: strip well suffix and simplify label
    long["group"] = long["raw_col"].apply(_clean_incucyte_group_name)

    # Replicate from within-group well order
    # Example: D1,D2,D3 -> R1,R2,R3 ; D4,D5,D6 -> R1,R2,R3
    long["well_id"] = long["raw_col"].apply(_extract_well_id)

    # Compute replicate index within each group based on original column order
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

    long["time"] = _coerce_time(long["time"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["group"] = long["group"].astype(str)
    long["replicate"] = long["replicate"].fillna("R1").astype(str)

    # Preserve group order from the file header
    group_order = pd.unique(col_map["group"]).tolist()
    long["group"] = pd.Categorical(long["group"], categories=group_order, ordered=True)

    long = long.dropna(subset=["time", "value"])

    return long[["time", "group", "replicate", "value"]].reset_index(drop=True)


def read_incucyte_csv(path_or_buffer) -> pd.DataFrame:
    """
    Read any of:
      A) Standard wide CSV: 'time' + one column per group
         Optional replicates via suffixes: A_R1, A_R2, B_R1 ...
      B) Standard tidy CSV: time, group, replicate, value
      C) Incucyte text export: metadata row + tab-delimited well columns

    Returns tidy DataFrame with columns: time, group, replicate, value
    """
    # --- First: check if this is an Incucyte text export ---
    if hasattr(path_or_buffer, "getvalue"):
        text = _read_text_buffer(path_or_buffer)
        if _detect_incucyte_export(text):
            return parse_incucyte_export(io.BytesIO(path_or_buffer.getvalue()))
        file_bytes = path_or_buffer.getvalue()
        csv_io = io.BytesIO(file_bytes)
    else:
        with open(path_or_buffer, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if _detect_incucyte_export(text):
            return parse_incucyte_export(path_or_buffer)
        csv_io = path_or_buffer

    # --- Otherwise: normal CSV/Tidy parsing ---
    df = pd.read_csv(csv_io)
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

        m = long["col"].astype(str).str.extract(r"^(.*)_(R\d+|Rep\d+|rep\d+)$")
        has_rep = m.notna().all(axis=1)

        long["group"] = np.where(has_rep, m[0], long["col"].astype(str))
        long["replicate"] = np.where(has_rep, m[1], "R1")

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
        "Could not detect supported format. Need either:\n"
        "  Wide CSV: 'time' + one column per group\n"
        "  Tidy CSV: time, group, (replicate), value\n"
        "  Incucyte export TXT/TSV: metadata row + Date Time / Elapsed / well columns"
    )


def aggregate_stats(df: pd.DataFrame, interval_hours: float = None) -> pd.DataFrame:
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
You can either **upload a CSV / TXT / TSV** *or* **enter the data manually**.

Supported formats:

- **Tidy**: `time, group, replicate, value`
- **Wide**: `time` + one column per group or replicate
- **Incucyte export TXT/TSV**: metadata row + `Date Time`, `Elapsed`, and well columns
"""
)

input_mode = st.radio(
    "How do you want to provide data?",
    ["Upload file", "Enter data manually"],
    index=0,
)

tidy = None

if input_mode == "Upload file":
    uploaded = st.file_uploader("Upload CSV / TXT / TSV", type=["csv", "txt", "tsv"])

    if uploaded is not None:
        try:
            tidy = read_incucyte_csv(uploaded)
            st.success("File parsed successfully.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.info("Upload a file to begin.")

else:
    st.markdown("### Enter your data")
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

if tidy is not None and not tidy.empty:
    st.subheader("Parsed data (tidy format)")
    st.dataframe(tidy.head())

    groups = pd.unique(tidy["group"].astype(str)).tolist()
    default_colors = make_color_list(len(groups))

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

    st.sidebar.markdown("### Groups")
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

    stats = aggregate_stats(tidy, interval_hours=interval_hours)

    stats = stats.merge(group_df, on="group", how="left", validate="many_to_one")
    tidy_merged = tidy.merge(group_df, on="group", how="left", validate="many_to_one")

    error_col = None
    plot_title_suffix = ""
    if error_choice == "SD":
        error_col = "sd"
        plot_title_suffix = " ± SD"
    elif error_choice == "SEM":
        error_col = "sem"
        plot_title_suffix = " ± SEM"

    fig, ax = plt.subplots(figsize=(8, 5))

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

    st.subheader("Summary table")
    st.dataframe(
        stats[["group", "display_name", "time", "mean", "sd", "sem", "n"]]
        .sort_values(["group", "time"])
        .reset_index(drop=True)
    )
