"""Builds a written report from analyzed results.

Every section is optional, so the person using the tool chooses what goes into the report.
The wording is generated from fixed templates filled in with the numbers, so the same data
always produces the same report.
"""

import html
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Optional

import markdown
import pandas as pd

from cleaning import CleaningResult
from insights import sentiment_by_group, sentiment_over_time, top_reasons
from sentiment import (
    LABELS,
    NEGATIVE_THRESHOLD,
    POSITIVE_THRESHOLD,
    RESERVED_COLUMNS,
    accuracy_against,
    normalize_labels,
)

# key -> heading shown in the report (and label shown in the dashboard)
SECTIONS = {
    "overview": "Overview",
    "data_quality": "Data quality",
    "cleaning": "Data cleaning",
    "breakdown": "Sentiment breakdown",
    "over_time": "Sentiment over time",
    "by_group": "Sentiment by group",
    "reasons": "Top reasons for negative texts",
    "words": "Most common words",
    "examples": "Example texts",
    "accuracy": "Accuracy check",
    "method": "Method",
}

MIN_PERIOD_TEXTS = 3     # periods with fewer texts are not used to name a "highest" period
MIN_GROUP_TEXTS = 30     # groups with fewer texts are treated with caution
MAX_TABLE_GROUPS = 15
MAX_TABLE_PERIODS = 12

STOPWORDS = set(
    """
    the and you for are was with this that have has had but not your from they will can just get got
    its our out all been when what about would there their them then than one now how who why did
    does doing were because into over after before again very much more most some any only also
    too let make made going gone say said see saw want need still even while here those these
    i'm it's don't can't didn't i've i'll we're you're isn't wasn't won't
    """.split()
)

METHOD_TEXT = (
    "1. The file is checked and, if cleaning is switched on, tidied first (see the data cleaning section).\n"
    "2. Links, @mentions and hashtag symbols are removed from each text.\n"
    "3. Each text is scored from -1 (very negative) to +1 (very positive) with VADER, "
    "a rule-based sentiment tool.{customer}\n"
    f"4. A score of {POSITIVE_THRESHOLD} or higher is labelled positive, a score of "
    f"{NEGATIVE_THRESHOLD} or lower is labelled negative, and anything in between is neutral.\n"
    "5. The wording of this report is generated from templates filled in with the numbers above."
)


@dataclass
class ReportInputs:
    """Extra information the report can use. Column names must be the names they have in the
    analyzed data. Anything left as None simply means that section is not available."""

    group_column: Optional[str] = None    # compare sentiment between these groups
    reason_column: Optional[str] = None   # why a text is negative (for example a complaint reason)
    label_column: Optional[str] = None    # an existing positive/neutral/negative label to check against
    date_column: Optional[str] = None     # when each text was written
    quality: Optional[dict] = None        # result of cleaning.assess_quality
    cleaning: Optional[CleaningResult] = None
    customer_messages: bool = False       # whether the customer-message word list was used


def pretty_name(column) -> str:
    """'user_group' -> 'user group' for use in headings and sentences."""
    name = str(column)
    prefix = "original_"
    if name.startswith(prefix) and name[len(prefix):] in RESERVED_COLUMNS:
        name = name[len(prefix):]  # a file column that had to be renamed during analysis
    return name.replace("_", " ").strip()


def available_sections(df: pd.DataFrame, inputs: Optional[ReportInputs] = None) -> dict:
    """Which sections can be built from this data and these inputs."""
    inputs = inputs or ReportInputs()

    def has(column) -> bool:
        return column is not None and column in df.columns

    avail = {key: True for key in SECTIONS}
    avail["data_quality"] = inputs.quality is not None
    avail["cleaning"] = inputs.cleaning is not None
    avail["by_group"] = has(inputs.group_column) and bool(df[inputs.group_column].notna().any())
    avail["reasons"] = has(inputs.reason_column)
    avail["accuracy"] = has(inputs.label_column) and bool(
        normalize_labels(df[inputs.label_column]).notna().any()
    )
    avail["over_time"] = (
        has(inputs.date_column) and len(sentiment_over_time(df, inputs.date_column)[0]) >= 2
    )
    return avail


# ---------- small helpers ----------

def _pct(part, total) -> str:
    return f"{part / total * 100:.1f}%" if total else "0.0%"


def _esc(value) -> str:
    return html.escape(str(value), quote=False)


def _cell(value) -> str:
    return _esc(value).replace("|", "\\|")


def _md_table(headers, rows) -> str:
    lines = [
        "| " + " | ".join(_cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(lines)


def _shorten(text, limit=220) -> str:
    text = " ".join(str(text).split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return _esc(text)


def _date(timestamp) -> str:
    return timestamp.strftime("%d %b %Y")


# ---------- one function per section ----------

def _overview(df: pd.DataFrame, inputs: ReportInputs) -> str:
    total = len(df)
    counts = df["sentiment"].value_counts()
    return (
        f"This report covers **{total:,}** texts. "
        f"{_pct(counts.get('negative', 0), total)} were negative, "
        f"{_pct(counts.get('neutral', 0), total)} were neutral and "
        f"{_pct(counts.get('positive', 0), total)} were positive. "
        f"The most common sentiment was **{counts.idxmax()}**."
    )


def _data_quality(df: pd.DataFrame, inputs: ReportInputs) -> str:
    q = inputs.quality
    rows = [
        ("Rows", f"{q['rows']:,}"),
        ("Columns", f"{q['columns']:,}"),
        ("Exact duplicate rows", f"{q['duplicate_rows']:,}"),
        ("Rows with no text", f"{q['blank_text']:,}"),
        ("Texts that repeat an earlier text", f"{q['repeated_texts']:,}"),
        ("Average words per text", f"{q['avg_words']:.1f}"),
    ]
    parts = ["Checked on the uploaded file, before any cleaning or filtering.", _md_table(["Check", "Result"], rows)]
    missing = q["missing"]
    if len(missing):
        listed = ", ".join(f"{_esc(col)} ({n:,})" for col, n in missing.head(8).items())
        more = f" and {len(missing) - 8} more" if len(missing) > 8 else ""
        parts.append(f"Columns with missing values: {listed}{more}.")
    if q["warnings"]:
        parts.append("**Warnings:**\n\n" + "\n".join(f"- {_esc(w)}" for w in q["warnings"]))
    else:
        parts.append("No problems were found with the text column.")
    return "\n\n".join(parts)


def _cleaning(df: pd.DataFrame, inputs: ReportInputs) -> str:
    result = inputs.cleaning
    removed = result.rows_before - result.rows_after
    lead = (
        f"The file had **{result.rows_before:,}** rows before cleaning and "
        f"**{result.rows_after:,}** after ({removed:,} removed)."
    )
    if not result.log:
        return lead + " No cleaning steps were switched on."
    rows = [(e["step"], f"{e['affected']:,}", e["detail"]) for e in result.log]
    return lead + "\n\n" + _md_table(["Step", "Affected", "What happened"], rows)


def _breakdown(df: pd.DataFrame, inputs: ReportInputs) -> str:
    total = len(df)
    counts = df["sentiment"].value_counts()
    rows = [
        (label.capitalize(), f"{counts.get(label, 0):,}", _pct(counts.get(label, 0), total))
        for label in LABELS
    ]
    return _md_table(["Sentiment", "Texts", "Share"], rows)


def _share(table: pd.DataFrame, column: str) -> float:
    """Weighted share (%) of one sentiment across several periods."""
    return float((table[column] * table["texts"]).sum() / table["texts"].sum())


def _over_time(df: pd.DataFrame, inputs: ReportInputs) -> str:
    table, unit = sentiment_over_time(df, inputs.date_column)
    parts = []
    intro = (
        f"The texts run from **{_date(table.index.min())}** to **{_date(table.index.max())}**, "
        f"grouped by {unit} ({len(table)} {unit}s)."
    )
    undated = len(df) - int(table["texts"].sum())
    if undated > 0:
        intro += f" {undated:,} texts had no readable date and are not included here."
    parts.append(intro)

    usable = table[table["texts"] >= MIN_PERIOD_TEXTS]
    if usable.empty:
        parts.append("Each period has too few texts to compare, so read these numbers with care.")
    else:
        worst, best = usable["negative"].idxmax(), usable["positive"].idxmax()
        parts.append(
            f"The {unit} starting **{_date(worst)}** had the highest share of negative texts "
            f"({usable.loc[worst, 'negative']:.1f}% of {int(usable.loc[worst, 'texts']):,}). "
            f"The {unit} starting **{_date(best)}** had the highest share of positive texts "
            f"({usable.loc[best, 'positive']:.1f}% of {int(usable.loc[best, 'texts']):,})."
        )
        if len(usable) >= 4:
            half = len(usable) // 2
            early, late = _share(usable.iloc[:half], "negative"), _share(usable.iloc[half:], "negative")
            if abs(late - early) < 1:
                trend = f"stayed about the same ({early:.1f}% then {late:.1f}%)"
            else:
                trend = f"{'rose' if late > early else 'fell'} from {early:.1f}% to {late:.1f}%"
            parts.append(
                f"Comparing the first half of this period with the second half, the share of "
                f"negative texts {trend}."
            )

    if len(table) <= MAX_TABLE_PERIODS:
        rows = [
            (_date(period), f"{int(r['texts']):,}", f"{r['positive']:.1f}%", f"{r['neutral']:.1f}%", f"{r['negative']:.1f}%")
            for period, r in table.iterrows()
        ]
        parts.append(_md_table([f"Start of {unit}", "Texts", "Positive", "Neutral", "Negative"], rows))
    return "\n\n".join(parts)


def _by_group(df: pd.DataFrame, inputs: ReportInputs) -> str:
    table = sentiment_by_group(df, inputs.group_column)
    if table.empty:
        return "There are no groups to compare."

    shown, note = table, ""
    if len(table) > MAX_TABLE_GROUPS:
        shown = table.nlargest(MAX_TABLE_GROUPS, "texts").sort_values("negative", ascending=False)
        note = f"The table shows the {MAX_TABLE_GROUPS} groups with the most texts, out of {len(table)}."
    rows = [
        (name, f"{int(r['texts']):,}", f"{r['positive']:.1f}%", f"{r['neutral']:.1f}%", f"{r['negative']:.1f}%")
        for name, r in shown.iterrows()
    ]
    heading = pretty_name(inputs.group_column).capitalize()
    table_md = _md_table([heading, "Texts", "Positive", "Neutral", "Negative"], rows)
    if len(table) < 2:
        return table_md

    pool = table[table["texts"] >= MIN_GROUP_TEXTS]
    if len(pool) < 2:
        pool = table
    worst, best = pool["negative"].idxmax(), pool["positive"].idxmax()
    text = (
        f"**{_esc(worst)}** has the highest share of negative texts ({pool.loc[worst, 'negative']:.1f}%). "
        f"**{_esc(best)}** has the highest share of positive texts ({pool.loc[best, 'positive']:.1f}%)."
    )
    if pool["texts"].min() < MIN_GROUP_TEXTS:
        text += " Groups with few texts should be read with care, because small numbers can swing the percentages."
    parts = [text] + ([note] if note else []) + [table_md]
    return "\n\n".join(parts)


def _reasons(df: pd.DataFrame, inputs: ReportInputs) -> str:
    reasons = top_reasons(df, inputs.reason_column)
    if reasons.empty:
        return "No reasons were recorded for the texts classified as negative."
    total = reasons.sum()
    top = reasons.head(5)
    rows = [(reason, f"{n:,}", _pct(n, total)) for reason, n in top.items()]
    lead = (
        f"The most common reason was **{_esc(top.index[0])}** "
        f"({_pct(top.iloc[0], total)} of the negative texts that have a recorded reason)."
    )
    return lead + "\n\n" + _md_table(["Reason", "Texts", "Share of recorded reasons"], rows)


def _top_words(texts, n=8):
    counter = Counter()
    for text in texts:
        counter.update(
            w
            for w in re.findall(r"[a-z']+", str(text).lower())
            if len(w) > 2 and w not in STOPWORDS
        )
    return counter.most_common(n)


def _words(df: pd.DataFrame, inputs: ReportInputs) -> str:
    items = []
    for label in ("negative", "positive"):
        top = _top_words(df.loc[df["sentiment"] == label, "clean_text"])
        if top:
            words = ", ".join(f"{_esc(w)} ({c})" for w, c in top)
            items.append(f"- **{label.capitalize()} texts:** {words}")
    if not items:
        return "There were not enough texts to find common words."
    return "The words used most often in each group (with how many times they appear):\n\n" + "\n".join(items)


def _examples(df: pd.DataFrame, inputs: ReportInputs, n: int = 3) -> str:
    parts = []
    for label, ascending in (("negative", True), ("positive", False)):
        subset = df[df["sentiment"] == label].sort_values("score", ascending=ascending).head(n)
        if subset.empty:
            continue
        parts.append(f"**Most {label}:**")
        parts.append("\n\n".join(f"> {_shorten(t)}" for t in subset["clean_text"]))
    return "\n\n".join(parts) if parts else "There were no texts to show."


def _accuracy(df: pd.DataFrame, inputs: ReportInputs) -> str:
    labels = normalize_labels(df[inputs.label_column])
    known = labels.notna()
    if not known.any():
        return "The chosen column has no positive, neutral or negative labels to compare against."
    acc = accuracy_against(df, inputs.label_column)
    conf = pd.crosstab(labels[known], df.loc[known, "sentiment"]).reindex(
        index=LABELS, columns=LABELS, fill_value=0
    )
    rows = [(f"Column says {row}", *[f"{conf.loc[row, col]:,}" for col in LABELS]) for row in LABELS]
    table = _md_table(["", "Tool says positive", "Tool says neutral", "Tool says negative"], rows)
    text = (
        f"The tool agreed with the labels in the **{_esc(pretty_name(inputs.label_column))}** column on "
        f"**{acc:.1%}** of {int(known.sum()):,} texts. "
        "Some disagreement is expected: the tool scores individual words, so it can miss sarcasm and context."
    )
    return text + "\n\n" + table


def _method(df: pd.DataFrame, inputs: ReportInputs) -> str:
    note = (
        " The customer-message setting was on: a list of common complaint words (such as cancelled, "
        "delayed, lost) was added, and \"help\" and \"please\" were treated as neutral."
        if inputs.customer_messages
        else ""
    )
    return METHOD_TEXT.replace("{customer}", note)


BUILDERS = {
    "overview": _overview,
    "data_quality": _data_quality,
    "cleaning": _cleaning,
    "breakdown": _breakdown,
    "over_time": _over_time,
    "by_group": _by_group,
    "reasons": _reasons,
    "words": _words,
    "examples": _examples,
    "accuracy": _accuracy,
    "method": _method,
}


# ---------- public functions ----------

def generate_report(
    df: pd.DataFrame,
    sections=None,
    title: str = "Sentiment Analysis Report",
    filters_note: str = "",
    inputs: Optional[ReportInputs] = None,
) -> str:
    """Return the report as Markdown.

    df must already be analyzed (it needs the sentiment, score and clean_text columns).
    sections is a list of keys from SECTIONS; None means every section.
    """
    inputs = inputs or ReportInputs()
    lines = [f"# {_esc(title)}", "", f"*Generated on {date.today():%d %B %Y}*"]
    if filters_note:
        lines += ["", f"*{_esc(filters_note)}*"]

    if df.empty:
        lines += ["", "There is no data to report on."]
        return "\n".join(lines) + "\n"

    wanted = set(sections) if sections is not None else set(SECTIONS)
    avail = available_sections(df, inputs)
    chosen = [key for key in SECTIONS if key in wanted and avail[key]]
    skipped = [SECTIONS[key] for key in SECTIONS if key in wanted and not avail[key]]

    if not chosen:
        lines += ["", "No sections were selected."]

    for key in chosen:
        heading = (
            f"Sentiment by {_esc(pretty_name(inputs.group_column))}" if key == "by_group" else SECTIONS[key]
        )
        lines += ["", f"## {heading}", "", BUILDERS[key](df, inputs)]

    if skipped:
        lines += [
            "",
            "---",
            "",
            "*Left out because the information they need was not available: " + ", ".join(skipped) + ".*",
        ]
    return "\n".join(lines) + "\n"


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: #1f2933;
         max-width: 780px; margin: 2rem auto; padding: 0 1.25rem; line-height: 1.6; }
  h1 { font-size: 1.9rem; margin-bottom: 0.2rem; }
  h2 { font-size: 1.25rem; margin-top: 2rem; border-bottom: 1px solid #d9dee3; padding-bottom: 0.3rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.95rem; }
  th, td { text-align: left; padding: 0.45rem 0.7rem; border-bottom: 1px solid #e4e7eb; }
  th { background: #f4f6f8; }
  blockquote { margin: 0.6rem 0; padding: 0.3rem 1rem; border-left: 4px solid #c9d1d9; color: #3e4c59; }
  hr { border: none; border-top: 1px solid #d9dee3; margin: 2rem 0 1rem; }
  @media print { body { margin: 0; max-width: none; } }
</style>
</head>
<body>
__BODY__
</body>
</html>
"""


def markdown_to_html(md_text: str, title: str = "Sentiment Analysis Report") -> str:
    """Wrap the Markdown report in a simple, printable HTML page."""
    body = markdown.markdown(md_text, extensions=["tables"])
    return _HTML_TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__BODY__", body)
