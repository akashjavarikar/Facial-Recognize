#python
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------- CONFIGURATION --------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_DIR / "logs"

st.set_page_config(
    page_title="Facial Expression Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Facial Expression Recognition Dashboard")
st.caption("Computer vision analytics and prediction history")

# -------------------- LOAD SESSION FILES --------------------

if not LOG_DIR.exists():
    st.error(f"Logs folder not found: {LOG_DIR}")
    st.info("Run the webcam script first to generate a CSV session.")
    st.stop()

csv_files = sorted(
    LOG_DIR.glob("session_*.csv"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if not csv_files:
    st.warning("No session CSV files found.")
    st.info(
        "Run python .\\src\\webcam_emotion.py, use the camera, "
        "then press Q to finish the session."
    )
    st.stop()

selected_file = st.selectbox(
    "Select webcam session",
    csv_files,
    format_func=lambda path: path.name,
)

# Read CSV and allow refresh from Streamlit's toolbar.
data = pd.read_csv(selected_file)

required_columns = [
    "timestamp",
    "face_id",
    "expression",
    "confidence_percent",
]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    st.error(f"CSV is missing required columns: {missing_columns}")
    st.info("Use the matching webcam_emotion.py code.")
    st.stop()

data["timestamp"] = pd.to_datetime(
    data["timestamp"], errors="coerce"
)

data["confidence_percent"] = pd.to_numeric(
    data["confidence_percent"], errors="coerce"
)

data["face_id"] = data["face_id"].astype(str)

for column in ["fps", "inference_latency_ms"]:
    if column in data.columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

data = data.dropna(
    subset=["timestamp", "confidence_percent"]
)

if data.empty:
    st.warning("This session has no valid prediction records.")
    st.stop()

# -------------------- SIDEBAR FILTERS --------------------

st.sidebar.header("Dashboard Filters")

available_expressions = sorted(
    data["expression"].dropna().astype(str).unique().tolist()
)

available_faces = sorted(
    data["face_id"].dropna().unique().tolist()
)

selected_expressions = st.sidebar.multiselect(
    "Select expressions",
    available_expressions,
    default=available_expressions,
)

selected_faces = st.sidebar.multiselect(
    "Select face IDs",
    available_faces,
    default=available_faces,
)

filtered = data[
    data["expression"].astype(str).isin(selected_expressions)
    & data["face_id"].isin(selected_faces)
].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# -------------------- SUMMARY METRICS --------------------

st.subheader("Session Overview")

total_records = len(filtered)
unique_faces = filtered["face_id"].nunique()
avg_confidence = filtered["confidence_percent"].mean()

avg_fps = None
if "fps" in filtered.columns:
    valid_fps = filtered["fps"].dropna()
    if not valid_fps.empty:
        avg_fps = valid_fps.mean()

avg_latency = None
if "inference_latency_ms" in filtered.columns:
    valid_latency = filtered["inference_latency_ms"].dropna()
    if not valid_latency.empty:
        avg_latency = valid_latency.mean()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Prediction Records", f"{total_records:,}")
col2.metric("Unique Face IDs", unique_faces)
col3.metric("Average Confidence", f"{avg_confidence:.1f}%")
col4.metric(
    "Average FPS",
    f"{avg_fps:.1f}" if avg_fps is not None else "N/A",
)
col5.metric(
    "Average Latency",
    f"{avg_latency:.1f} ms" if avg_latency is not None else "N/A",
)

st.divider()

# -------------------- EXPRESSION DISTRIBUTION --------------------

distribution = (
    filtered["expression"]
    .value_counts()
    .rename_axis("Expression")
    .reset_index(name="Count")
)

left, right = st.columns(2)

with left:
    st.subheader("Expression Distribution")

    fig_bar = px.bar(
        distribution,
        x="Expression",
        y="Count",
        color="Expression",
        title="Recorded predictions by expression",
    )

    st.plotly_chart(
        fig_bar,
        width="stretch",
        config={},
    )

with right:
    st.subheader("Expression Share")

    fig_pie = px.pie(
        distribution,
        names="Expression",
        values="Count",
        hole=0.4,
        title="Share of recorded predictions",
    )

    st.plotly_chart(
        fig_pie,
        width="stretch",
        config={},
    )

# -------------------- CONFIDENCE HISTORY --------------------

st.subheader("Confidence History")

confidence_data = filtered.sort_values("timestamp")

fig_confidence = px.line(
    confidence_data,
    x="timestamp",
    y="confidence_percent",
    color="face_id",
    markers=True,
    hover_data=["expression"],
    title="Prediction confidence over time",
)

fig_confidence.update_yaxes(title="Confidence (%)")
fig_confidence.update_xaxes(title="Time")

st.plotly_chart(
    fig_confidence,
    width="stretch",
    config={},
)

# -------------------- LATENCY HISTORY --------------------

if (
    "inference_latency_ms" in filtered.columns
    and filtered["inference_latency_ms"].notna().any()
):
    st.subheader("Inference Latency")

    latency_data = filtered.dropna(
        subset=["inference_latency_ms"]
    ).sort_values("timestamp")

    fig_latency = px.line(
        latency_data,
        x="timestamp",
        y="inference_latency_ms",
        markers=True,
        title="Recorded inference latency over time",
    )

    fig_latency.update_yaxes(title="Latency (ms)")

    st.plotly_chart(
        fig_latency,
        width="stretch",
        config={},
    )
else:
    st.info(
        "Latency data is not available in this session. "
        "Update the webcam script and record a new session to collect it."
    )

# -------------------- CLASS SCORES --------------------

score_columns = [
    column for column in filtered.columns
    if column.startswith("score_")
]

if score_columns:
    st.subheader("Individual Emotion Scores")

    selected_face = st.selectbox(
        "Choose a face to inspect",
        sorted(filtered["face_id"].unique().tolist()),
    )

    face_data = filtered[
        filtered["face_id"] == selected_face
    ].sort_values("timestamp")

    score_data = face_data.melt(
        id_vars=["timestamp", "face_id"],
        value_vars=score_columns,
        var_name="Expression",
        value_name="Score (%)",
    )

    score_data["Expression"] = (
        score_data["Expression"]
        .str.replace("score_", "", regex=False)
        .str.title()
    )

    fig_scores = px.line(
        score_data,
        x="timestamp",
        y="Score (%)",
        color="Expression",
        title=f"Emotion scores for Face {selected_face}",
    )

    st.plotly_chart(
        fig_scores,
        width="stretch",
        config={},
    )

# -------------------- RECORDED DATA --------------------

st.subheader("Recorded Predictions")

st.dataframe(
    filtered.sort_values("timestamp", ascending=False),
    width="stretch",
    hide_index=True,
)

# -------------------- DOWNLOAD CSV --------------------

csv_output = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered CSV",
    data=csv_output,
    file_name=f"filtered_{selected_file.name}",
    mime="text/csv",
)

st.caption(
    "Predictions are model estimates, not verified measurements "
    "of a person's actual emotional state."
)