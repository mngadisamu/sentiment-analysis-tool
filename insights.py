"""Shared calculations used by the dashboard, the script and the report."""

import pandas as pd

from cleaning import parse_dates
from sentiment import LABELS


def sentiment_by_group(df, group_column) -> pd.DataFrame:
    """For each group: number of texts and the percentage that are positive, neutral, negative.

    Sorted with the most negative group first.
    """
    counts = pd.crosstab(df[group_column], df["sentiment"]).reindex(columns=LABELS, fill_value=0)
    totals = counts.sum(axis=1)
    table = counts.div(totals, axis=0) * 100
    table.insert(0, "texts", totals)
    table.columns.name = None
    return table.sort_values("negative", ascending=False)


def top_reasons(df, reason_column, n=None) -> pd.Series:
    """How often each reason appears among the texts the tool labelled negative."""
    negatives = df.loc[df["sentiment"] == "negative", reason_column].dropna()
    counts = negatives.value_counts()
    return counts.head(n) if n else counts


def sentiment_over_time(df, date_column):
    """Percentage of positive, neutral and negative texts per day, week or month.

    The size of the time step is chosen from how long the data spans. Returns the table
    (one row per period) and the name of the step ("day", "week" or "month").
    """
    dates = parse_dates(df[date_column])
    keep = dates.notna()
    if not keep.any():
        return pd.DataFrame(columns=["texts", *LABELS]), "day"

    valid = dates[keep]
    span_days = (valid.max() - valid.min()).days
    if span_days <= 45:
        freq, unit = "D", "day"
    elif span_days <= 730:
        freq, unit = "W", "week"
    else:
        freq, unit = "M", "month"

    period = valid.dt.to_period(freq).dt.start_time
    counts = pd.crosstab(period, df.loc[keep, "sentiment"]).reindex(columns=LABELS, fill_value=0)
    totals = counts.sum(axis=1)
    table = counts.div(totals, axis=0) * 100
    table.insert(0, "texts", totals)
    table.index.name = "period"
    table.columns.name = None
    return table.sort_index(), unit
