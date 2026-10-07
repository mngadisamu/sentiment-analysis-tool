"""Streamlit dashboard version of the sentiment analysis tool.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from cleaning import (
    CleaningOptions,
    CleaningResult,
    assess_quality,
    clean_dataframe,
    guess_text_column,
    suggest_columns,
)
from insights import sentiment_by_group, sentiment_over_time, top_reasons
from report import (
    SECTIONS,
    ReportInputs,
    available_sections,
    generate_report,
    markdown_to_html,
    pretty_name,
)
from sentiment import LABELS, analyze_dataframe, reserved_name

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_tweets.csv"
COLORS = {"positive": "#2a9d8f", "neutral": "#8d99ae", "negative": "#e76f51"}
NONE = "(none)"
MAX_GROUPS_IN_CHART = 15

st.set_page_config(page_title="Sentiment Analysis Tool", layout="wide")


@st.cache_data
def check_quality(raw: pd.DataFrame, text_column: str) -> dict:
    return assess_quality(raw, text_column)


@st.cache_data(show_spinner="Cleaning and analyzing...")
def run_pipeline(raw, text_column, customer_messages, date_column, do_clean, trim, blanks, dup_rows, dup_text, categories, dates):
    if do_clean:
        options = CleaningOptions(
            trim_whitespace=trim,
            drop_blank_text=blanks,
            drop_duplicate_rows=dup_rows,
            drop_duplicate_text=dup_text,
            standardize_categories=categories,
            parse_dates=dates,
        )
        cleaning = clean_dataframe(raw, text_column, options, date_column)
    else:
        cleaning = CleaningResult(data=raw.copy(), log=[], rows_before=len(raw), rows_after=len(raw))
    return analyze_dataframe(cleaning.data, text_column, customer_messages), cleaning


def to_long(table: pd.DataFrame, id_name: str) -> pd.DataFrame:
    """One row per (period or group, sentiment) so Plotly can colour by sentiment."""
    return (
        table[LABELS]
        .rename_axis(id_name)
        .reset_index()
        .melt(id_vars=id_name, var_name="sentiment", value_name="percent")
    )


st.title("Sentiment Analysis Tool")
st.caption("Upload a CSV of texts, clean it, explore the sentiment, and download a written report.")

# ---------- Sidebar 1: data ----------
st.sidebar.header("1. Data")
uploaded = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as error:  # a bad file should give a message, not a crash
        st.error(f"Could not read that file as a CSV: {error}")
        st.stop()
elif SAMPLE_PATH.exists():
    raw = pd.read_csv(SAMPLE_PATH)
    st.sidebar.info("Showing the built-in sample. Upload your own CSV to replace it.")
else:
    st.info("Upload a CSV file to begin.")
    st.stop()

if raw.empty:
    st.error("That file has no rows.")
    st.stop()

columns = list(raw.columns)
default_text = "text" if "text" in columns else guess_text_column(raw)
text_column = st.sidebar.selectbox(
    "Column that contains the text", columns, index=columns.index(default_text)
)

# ---------- Sidebar 2: which column is which ----------
st.sidebar.header("2. Columns (optional)")
st.sidebar.caption("Guessed from your file. Change them if they are wrong, or choose (none).")
suggested = suggest_columns(raw, text_column)


def pick(label: str, key: str, help_text: str):
    options = [NONE] + [c for c in columns if c != text_column]
    default = suggested[key]
    index = options.index(default) if default in options else 0
    choice = st.sidebar.selectbox(label, options, index=index, help=help_text)
    return None if choice == NONE else choice


group_column = pick("Compare sentiment by", "group", "For example airline, product, branch or country.")
reason_column = pick("Reason for negative texts", "reason", "A column that explains complaints, such as a complaint reason.")
label_column = pick("Existing sentiment labels", "label", "A column that already says positive, neutral or negative. Used to check the tool's accuracy.")
date_column = pick("Date", "date", "When each text was written. Used for the sentiment over time chart.")

# ---------- Sidebar 3: cleaning ----------
st.sidebar.header("3. Cleaning")
do_clean = st.sidebar.checkbox("Clean the data before analysis", value=True)
trim = st.sidebar.checkbox("Tidy extra spaces", value=True, disabled=not do_clean)
categories = st.sidebar.checkbox("Make spelling and capitals consistent", value=True, disabled=not do_clean)
blanks = st.sidebar.checkbox("Remove rows with no text", value=True, disabled=not do_clean)
dup_rows = st.sidebar.checkbox("Remove duplicate rows", value=True, disabled=not do_clean)
dup_text = st.sidebar.checkbox("Also remove repeated texts", value=False, disabled=not do_clean)
dates = st.sidebar.checkbox("Read the date column as dates", value=True, disabled=not do_clean or date_column is None)

st.sidebar.subheader("Scoring")
customer_messages = st.sidebar.checkbox(
    "Customer messages mode",
    value=True,
    help="Adds common complaint words (cancelled, delayed, lost...) and treats 'help' and 'please' as neutral. "
    "Turn it off for text that is not complaints or support messages.",
)

quality = check_quality(raw, text_column)
analyzed, cleaning = run_pipeline(
    raw, text_column, customer_messages, date_column, do_clean, trim, blanks, dup_rows, dup_text, categories, dates
)

if analyzed.empty:
    st.error("Cleaning removed every row. Check the text column, or switch some cleaning options off.")
    st.stop()

# Columns called clean_text, sentiment or score in the file are renamed original_* during analysis
group_col, reason_col, label_col, date_col = (
    reserved_name(c) for c in (group_column, reason_column, label_column, date_column)
)

# ---------- Sidebar 4: filters ----------
st.sidebar.header("4. Filters")
filtered = analyzed
filters_note = ""

if group_col:
    values = sorted(analyzed[group_col].dropna().unique(), key=str)
    if len(values) <= 200:
        chosen = st.sidebar.multiselect(
            f"Include these {pretty_name(group_column)} values", values, default=values
        )
        if len(chosen) < len(values):
            filtered = analyzed[analyzed[group_col].isin(chosen)]
            names = [str(v) for v in chosen]
            shown = ", ".join(names[:10]) + (f" and {len(names) - 10} more" if len(names) > 10 else "")
            filters_note = f"Only these {pretty_name(group_column)} values are included: {shown}."
    else:
        st.sidebar.caption("Too many different values to filter by.")
else:
    st.sidebar.caption("Choose a column to compare by (section 2) to filter by it.")

if filtered.empty:
    st.warning("Nothing to show with the current filters. Select at least one option.")
    st.stop()

# ---------- Sidebar 5: report options ----------
st.sidebar.header("5. Report")
report_title = st.sidebar.text_input("Report title", "Sentiment Analysis Report")
inputs = ReportInputs(
    group_column=group_col,
    reason_column=reason_col,
    label_column=label_col,
    date_column=date_col,
    quality=quality,
    cleaning=cleaning if do_clean else None,
    customer_messages=customer_messages,
)
available = available_sections(filtered, inputs)
section_labels = dict(SECTIONS)
if group_column:
    section_labels["by_group"] = f"Sentiment by {pretty_name(group_column)}"

st.sidebar.write("Include these sections:")
selected = [
    key
    for key, label in section_labels.items()
    if available[key] and st.sidebar.checkbox(label, value=True, key=f"section_{key}")
]
unavailable = [label for key, label in section_labels.items() if not available[key]]
if unavailable:
    st.sidebar.caption("Not available for this data: " + ", ".join(unavailable) + ".")

# ---------- Main area ----------
tab_dashboard, tab_quality, tab_report, tab_data = st.tabs(
    ["Dashboard", "Data quality & cleaning", "Report", "Data"]
)

with tab_dashboard:
    total = len(filtered)
    counts = filtered["sentiment"].value_counts()

    metrics = st.columns(4)
    metrics[0].metric("Texts analyzed", f"{total:,}")
    for column, label in zip(metrics[1:], LABELS):
        column.metric(label.capitalize(), f"{counts.get(label, 0) / total:.1%}")

    sentiment_df = pd.DataFrame(
        {"sentiment": LABELS, "texts": [int(counts.get(label, 0)) for label in LABELS]}
    )
    fig = px.bar(
        sentiment_df,
        x="sentiment",
        y="texts",
        color="sentiment",
        color_discrete_map=COLORS,
        title="Positive, neutral and negative texts",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

    if available["over_time"]:
        over_time, unit = sentiment_over_time(filtered, date_col)
        fig = px.line(
            to_long(over_time, "period"),
            x="period",
            y="percent",
            color="sentiment",
            markers=True,
            color_discrete_map=COLORS,
            category_orders={"sentiment": LABELS},
            title=f"Sentiment over time (% of each {unit}'s texts)",
        )
        fig.update_yaxes(title="% of texts", rangemode="tozero")
        st.plotly_chart(fig)

    if available["by_group"]:
        by_group = sentiment_by_group(filtered, group_col)
        title = f"Sentiment by {pretty_name(group_column)} (% of each group's texts)"
        if len(by_group) > MAX_GROUPS_IN_CHART:
            by_group = by_group.nlargest(MAX_GROUPS_IN_CHART, "texts")
            title = f"Sentiment by {pretty_name(group_column)}: the {MAX_GROUPS_IN_CHART} groups with the most texts"
        long = to_long(by_group, "group")
        long["group"] = long["group"].astype(str)
        fig = px.bar(
            long,
            x="group",
            y="percent",
            color="sentiment",
            color_discrete_map=COLORS,
            category_orders={"sentiment": LABELS},
            title=title,
        )
        st.plotly_chart(fig)

    if available["reasons"]:
        reasons = top_reasons(filtered, reason_col, 8)
        if not reasons.empty:
            reasons_df = pd.DataFrame({"reason": reasons.index.astype(str), "texts": reasons.values})
            fig = px.bar(
                reasons_df.iloc[::-1],
                x="texts",
                y="reason",
                orientation="h",
                title="Top reasons for negative texts",
            )
            fig.update_traces(marker_color=COLORS["negative"])
            st.plotly_chart(fig)

with tab_quality:
    st.subheader("Your file, before cleaning")
    cols = st.columns(4)
    cols[0].metric("Rows", f"{quality['rows']:,}")
    cols[1].metric("Exact duplicate rows", f"{quality['duplicate_rows']:,}")
    cols[2].metric("Rows with no text", f"{quality['blank_text']:,}")
    cols[3].metric("Average words per text", f"{quality['avg_words']:.1f}")
    for warning in quality["warnings"]:
        st.warning(warning)
    if not quality["warnings"]:
        st.success("No problems were found with the text column.")
    if len(quality["missing"]):
        st.write("**Columns with missing values**")
        st.dataframe(quality["missing"].rename("Missing values").to_frame())

    st.subheader("Cleaning")
    if not do_clean:
        st.info("Cleaning is switched off. The data is analyzed exactly as it was uploaded.")
    else:
        cols = st.columns(3)
        cols[0].metric("Rows before", f"{cleaning.rows_before:,}")
        cols[1].metric("Rows after", f"{cleaning.rows_after:,}")
        cols[2].metric("Rows removed", f"{cleaning.rows_before - cleaning.rows_after:,}")
        if cleaning.log:
            log_df = pd.DataFrame(cleaning.log).rename(
                columns={"step": "Step", "affected": "Affected", "detail": "What happened"}
            )
            st.dataframe(log_df, hide_index=True)
        st.download_button(
            "Download cleaned data (.csv)",
            cleaning.data.to_csv(index=False),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )

with tab_report:
    report_md = generate_report(filtered, selected, report_title, filters_note, inputs)
    report_html = markdown_to_html(report_md, report_title)

    left, right = st.columns(2)
    left.download_button(
        "Download report (.html)", report_html, file_name="sentiment_report.html", mime="text/html"
    )
    right.download_button(
        "Download report (.md)", report_md, file_name="sentiment_report.md", mime="text/markdown"
    )
    st.caption("Tip: open the .html file in a browser and choose Print, then Save as PDF.")
    st.divider()
    st.markdown(report_md)

with tab_data:
    st.download_button(
        "Download results (.csv)",
        filtered.to_csv(index=False),
        file_name="sentiment_results.csv",
        mime="text/csv",
    )
    st.dataframe(filtered)
