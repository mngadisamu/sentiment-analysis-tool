"""Core sentiment logic, shared by the script (analyze.py) and the dashboard (app.py)."""

import html
import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# "Customer messages" setting: VADER was built for general social media text. In complaints and
# support messages it misses many negative words (a "cancelled" or "delayed" flight is not
# negative to VADER) and over-counts polite words ("please", "help"). This list adds the common
# complaint words and sets the polite ones to neutral. Anything not listed keeps VADER's own score.
COMPLAINT_WORDS = {
    word: -1.8
    for word in (
        "cancelled canceled cancellation cancelling delayed delay delays lost stuck stranded "
        "waiting wait waited hold late missed rude never refund rebook unacceptable ridiculous "
        "hours nobody still again broken damaged not ignored unhelpful overbooked sitting"
    ).split()
}
COMPLAINT_WORDS.update({"flightled": -2.2, "flighted": -2.2})  # airline slang for a cancelled flight
NEUTRAL_POLITE_WORDS = {"help": 0.0, "please": 0.0}

_customer_analyzer = SentimentIntensityAnalyzer()
_customer_analyzer.lexicon.update(COMPLAINT_WORDS)
_customer_analyzer.lexicon.update(NEUTRAL_POLITE_WORDS)

LABELS = ["positive", "neutral", "negative"]

# VADER's recommended thresholds on the compound score (-1 to +1)
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# Columns this tool adds. If the uploaded data already has a column with one of these
# names, the original is kept under a new name (original_<name>) so nothing is overwritten.
RESERVED_COLUMNS = ("clean_text", "sentiment", "score")


def reserved_name(column):
    """The name a column has after analysis (only renamed if it clashes with ours)."""
    if column is not None and column in RESERVED_COLUMNS:
        return f"original_{column}"
    return column


def clean_text(text) -> str:
    """Remove links, @mentions and hashtag symbols, turn &amp; style codes into real characters,
    and tidy whitespace."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.replace("#", "")
    return re.sub(r"\s+", " ", text).strip()


def classify(text: str, customer_messages: bool = False) -> tuple[str, float]:
    """Return (label, compound_score) for one piece of text."""
    analyzer = _customer_analyzer if customer_messages else _analyzer
    score = analyzer.polarity_scores(text)["compound"]
    if score >= POSITIVE_THRESHOLD:
        return "positive", score
    if score <= NEGATIVE_THRESHOLD:
        return "negative", score
    return "neutral", score


def analyze_dataframe(
    df: pd.DataFrame, text_column: str = "text", customer_messages: bool = True
) -> pd.DataFrame:
    """Add clean_text, sentiment and score columns to a copy of df.

    customer_messages=True uses the complaint word list (see COMPLAINT_WORDS); False is plain VADER.
    """
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
        )
    source = df[text_column]
    out = df.rename(
        columns={c: f"original_{c}" for c in RESERVED_COLUMNS if c in df.columns}
    ).copy()
    out["clean_text"] = source.apply(clean_text)
    results = out["clean_text"].apply(lambda t: classify(t, customer_messages))
    out["sentiment"] = results.apply(lambda r: r[0])
    out["score"] = results.apply(lambda r: r[1])
    return out


def sentiment_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage for each sentiment label."""
    counts = df["sentiment"].value_counts()
    summary = pd.DataFrame({"count": counts})
    summary["percent"] = (summary["count"] / summary["count"].sum() * 100).round(1)
    return summary


_LABEL_SYNONYMS = {
    "positive": "positive", "pos": "positive",
    "neutral": "neutral", "neu": "neutral",
    "negative": "negative", "neg": "negative",
}


def normalize_labels(series: pd.Series) -> pd.Series:
    """Turn labels like 'Positive' or 'neg' into positive / neutral / negative.

    Anything that is not a recognised sentiment label becomes blank.
    """
    return series.map(
        lambda v: _LABEL_SYNONYMS.get(str(v).strip().casefold()) if pd.notna(v) else None
    )


def accuracy_against(df: pd.DataFrame, label_column: str) -> float:
    """Share of rows where the tool agrees with an existing label column.

    Labels are compared ignoring capital letters and extra spaces, so 'Positive' matches 'positive'.
    Rows whose label is not a sentiment label are left out of the comparison.
    """
    labels = normalize_labels(df[label_column])
    known = labels.notna()
    if not known.any():
        return float("nan")
    return float((df.loc[known, "sentiment"] == labels[known]).mean())
