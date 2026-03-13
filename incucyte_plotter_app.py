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


def read_incucyte_csv(path_or_buffer) -> pd.DataFrame:
    """
    Read either:
      A) WIDE format: 'time' + one column per group (e.g., Time,A,B,C)
         Optional replicates via suffixes: A_R1, A_R2, B_R1 ...
      B) TIDY format: time, group, replicate, value

    Returns a tidy DataFrame with columns: time, group, replicate, value
    """
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
        "  Tidy: time, group, (replicate), value"
    )


def aggregate_mean_sd(df: pd.DataFrame, interval_hours: float = None) -> pd.DataFrame:
    """
    Aggregate replicates to group-level mean ± SD.
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
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

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

    # Unique groups
    groups = sorted(pd.unique(tidy["group"].astype(str)).tolist())

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

    error_choice = st.sidebar.selectbox("Error bars", ["SD", "None"], index=0)

    smooth_step = st.sidebar.number_input(
        "Plot every Nth timepoint (visual smoothing)",
        min_value=1,
        value=1,
        step=1,
        help="1 = use all points; 2 = every 2nd; 3 = every 3rd; etc.",
    )

    # Editable group table for names & colours
    st.sidebar.markdown("### Groups")

    color_list = make_color_list(len(groups))

    group_df = pd.DataFrame(
        {
            "group": groups,
            "display_name": groups,
            "color": color_list,
        }
    )

    edited_group_df = st.sidebar.data_editor(
        group_df,
        num_rows="dynamic",
        use_container_width=True,
        key="group_editor",
    )

    # ---------- Compute stats ----------
    stats = aggregate_mean_sd(tidy, interval_hours=interval_hours)

    # Merge display names & colours
    stats = stats.merge(
        edited_group_df, on="group", how="left", validate="many_to_one"
    )
    tidy_merged = tidy.merge(
        edited_group_df, on="group", how="left", validate="many_to_one"
    )

    # ---------- Plot: mean ± SD ----------
    fig, ax = plt.subplots(figsize=(8, 5))

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
            linewidth=2,
        )

        if error_choice == "SD" and sub_plot["sd"].notna().any():
            ax.fill_between(
                sub_plot["time"],
                sub_plot["mean"] - sub_plot["sd"],
                sub_plot["mean"] + sub_plot["sd"],
                alpha=0.2,
                color=color,
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    st.subheader("Mean ± SD per group")
    st.pyplot(fig)

    buf_mean = io.BytesIO()
    fig.savefig(buf_mean, format="png", dpi=300, bbox_inches="tight")
    buf_mean.seek(0)
    st.download_button(
        "Download mean plot (PNG, 300 dpi)",
        data=buf_mean,
        file_name="incucyte_mean_sd.png",
        mime="image/png",
    )

    # ---------- Plot: replicate spaghetti ----------
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for (g, r), sub in tidy_merged.groupby(["group", "replicate"], sort=False):
        sub = sub.sort_values("time")
        name = sub["display_name"].iloc[0]
        color = sub["color"].iloc[0] if pd.notna(sub["color"].iloc[0]) else None
        ax2.plot(sub["time"], sub["value"], color=color, alpha=0.4, label=name)

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

    buf_spag = io.BytesIO()
    fig2.savefig(buf_spag, format="png", dpi=300, bbox_inches="tight")
    buf_spag.seek(0)
    st.download_button(
        "Download spaghetti plot (PNG, 300 dpi)",
        data=buf_spag,
        file_name="incucyte_spaghetti.png",
        mime="image/png",
    )

    # ---------- Summary table + CSV ----------
    st.subheader("Summary (mean ± SD)")
    st.dataframe(stats[["group", "display_name", "time", "mean", "sd", "n"]])

    csv_buffer = io.StringIO()
    stats.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download summary CSV",
        data=csv_buffer.getvalue(),
        file_name="incucyte_summary_mean_sd.csv",
        mime="text/csv",
    )
