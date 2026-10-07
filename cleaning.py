"""Data preparation: a quality check, cleaning steps and column suggestions.

Nothing here changes the original file. Cleaning always works on a copy and returns a log
that says exactly what was changed, so the numbers can be explained and trusted.
"""

import re
import warnings
from dataclasses import dataclass, field

import pandas as pd

from sentiment import normalize_labels

# Columns with more different values than this are not treated as categories
MAX_CATEGORIES = 100

DATE_NAME_HINTS = ("date", "created", "timestamp", "time", "posted", "published")
GROUP_NAME_HINTS = (
    "airline", "brand", "product", "category", "branch", "region", "country",
    "segment", "topic", "department", "source", "platform", "store",
)


@dataclass
class CleaningOptions:
    trim_whitespace: bool = True          # remove extra spaces in every text column
    drop_blank_text: bool = True          # remove rows with no text to analyze
    drop_duplicate_rows: bool = True      # remove rows that are exact copies of another row
    drop_duplicate_text: bool = False     # also remove rows that repeat the same text
    standardize_categories: bool = True   # "late flight" / "Late Flight " -> one spelling
    parse_dates: bool = True              # read the date column as real dates


@dataclass
class CleaningResult:
    data: pd.DataFrame
    log: list = field(default_factory=list)  # one dict per step: step, affected, detail
    rows_before: int = 0
    rows_after: int = 0


# ---------- small helpers ----------

def _is_textual(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _tidy_value(value):
    return " ".join(value.split()) if isinstance(value, str) else value


def _key(value) -> str:
    """Spelling-insensitive version of a value, used to spot 'Late Flight' vs 'late flight '."""
    return " ".join(str(value).split()).casefold()


def _differs(before: pd.Series, after: pd.Series) -> int:
    """How many cells changed (blank cells are ignored)."""
    return int(((before != after) & before.notna() & after.notna()).sum())


_TRAILING_OFFSET = re.compile(r"\s*(?:Z|[+-]\d{2}:?\d{2})$")


def _drop_offset(value):
    """'2015-02-24 16:20:00 -0800' -> '2015-02-24 16:20:00' (only values that include a time)."""
    if isinstance(value, str) and ":" in value:
        return _TRAILING_OFFSET.sub("", value.strip())
    return value


def parse_dates(series: pd.Series) -> pd.Series:
    """Read a column as dates. Anything that cannot be read becomes blank (NaT).

    Dates are kept as written: a time-zone offset at the end of a value (such as -0800) is
    ignored, so a tweet written at 4pm on the 24th stays on the 24th.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.tz_localize(None) if getattr(series.dt, "tz", None) is not None else series
    text = series.map(_drop_offset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(text, errors="coerce", utc=True, format="mixed")
        except (TypeError, ValueError):  # older pandas versions have no format="mixed"
            parsed = pd.to_datetime(text, errors="coerce", utc=True)
    return parsed.dt.tz_localize(None)


def _spelling_map(series: pd.Series) -> dict:
    """Map every minority spelling of a value to its most common spelling."""
    counts = series.dropna().value_counts()
    variants = {}
    for raw, n in counts.items():
        variants.setdefault(_key(raw), []).append((raw, n))
    mapping = {}
    for options in variants.values():
        if len(options) > 1:
            canonical = max(options, key=lambda item: item[1])[0]
            mapping.update({raw: canonical for raw, _ in options if raw != canonical})
    return mapping


# ---------- cleaning ----------

def clean_dataframe(df, text_column, options=None, date_column=None) -> CleaningResult:
    """Clean a copy of df and return it with a log of what was changed."""
    options = options or CleaningOptions()
    data = df.copy()
    rows_before = len(data)
    log = []

    text_columns = [c for c in data.columns if _is_textual(data[c])]

    if options.trim_whitespace:
        changed = 0
        for col in text_columns:
            tidy = data[col].map(_tidy_value)
            changed += _differs(data[col], tidy)
            data[col] = tidy
        log.append({
            "step": "Tidied extra spaces",
            "affected": changed,
            "detail": "Removed leading, trailing and double spaces in text columns.",
        })

    if options.standardize_categories:
        changed, examples = 0, []
        for col in text_columns:
            if col in (text_column, date_column):
                continue
            series = data[col]
            if series.nunique(dropna=True) > MAX_CATEGORIES:
                continue
            mapping = _spelling_map(series)
            if not mapping:
                continue
            counts = series.value_counts()
            changed += int(sum(counts.get(raw, 0) for raw in mapping))
            data[col] = series.map(lambda v, m=mapping: m.get(v, v))
            examples += [f"{col}: '{raw}' to '{canon}'" for raw, canon in list(mapping.items())[:2]]
        detail = "Made spelling and capital letters consistent within each category column."
        if examples:
            detail += " For example " + "; ".join(examples[:3]) + "."
        log.append({"step": "Standardized spelling", "affected": changed, "detail": detail})

    if options.drop_blank_text:
        as_text = data[text_column].map(lambda v: "" if pd.isna(v) else str(v).strip())
        blank = as_text.eq("")
        data = data.loc[~blank]
        log.append({
            "step": "Removed rows with no text",
            "affected": int(blank.sum()),
            "detail": "Rows where the text column was empty cannot be analyzed.",
        })

    if options.drop_duplicate_rows:
        duplicated = data.duplicated()
        data = data.loc[~duplicated]
        log.append({
            "step": "Removed duplicate rows",
            "affected": int(duplicated.sum()),
            "detail": "Rows that were exact copies of an earlier row (the first copy is kept).",
        })

    if options.drop_duplicate_text:
        repeated = data[text_column].map(_key).duplicated()
        data = data.loc[~repeated]
        log.append({
            "step": "Removed repeated texts",
            "affected": int(repeated.sum()),
            "detail": "Rows with the same text as an earlier row (the first one is kept).",
        })

    if options.parse_dates and date_column and date_column in data.columns:
        parsed = parse_dates(data[date_column])
        unreadable = int(parsed.isna().sum() - data[date_column].isna().sum())
        data[date_column] = parsed
        log.append({
            "step": "Read dates",
            "affected": int(parsed.notna().sum()),
            "detail": f"Read '{date_column}' as dates. {unreadable:,} value(s) could not be read and were left blank.",
        })

    data = data.reset_index(drop=True)
    return CleaningResult(data=data, log=log, rows_before=rows_before, rows_after=len(data))


# ---------- quality check ----------

def assess_quality(df, text_column) -> dict:
    """Check a file before cleaning and warn about anything that looks wrong."""
    rows = len(df)
    as_text = df[text_column].map(lambda v: "" if pd.isna(v) else str(v).strip())
    has_text = as_text.ne("")
    non_blank = as_text[has_text]
    blank = int((~has_text).sum())

    avg_words = float(non_blank.map(lambda s: len(s.split())).mean()) if len(non_blank) else 0.0
    numeric_share = (
        float(non_blank.str.fullmatch(r"[\d\s.,+\-]+").mean()) if len(non_blank) else 0.0
    )
    unique_texts = int(non_blank.str.casefold().nunique())
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    duplicate_rows = int(df.duplicated().sum())

    warns = []
    if rows == 0:
        warns.append("The file has no rows.")
    else:
        if blank / rows > 0.2:
            warns.append(f"{blank / rows:.0%} of the rows have no text in this column.")
        if len(non_blank) and avg_words < 3:
            warns.append(
                f"The texts are very short (about {avg_words:.1f} words on average). Sentiment "
                "works best on full sentences, so check this is the right column."
            )
        if numeric_share > 0.5:
            warns.append("Most values in this column are numbers, so it is probably not free text.")
        if len(non_blank) >= 50 and unique_texts < 0.1 * len(non_blank):
            warns.append(
                f"Only {unique_texts:,} different texts in {len(non_blank):,} rows. This looks like "
                "a category column, not free text."
            )
        if duplicate_rows / rows > 0.05:
            warns.append(f"{duplicate_rows / rows:.0%} of the rows are exact duplicates.")

    return {
        "rows": rows,
        "columns": len(df.columns),
        "duplicate_rows": duplicate_rows,
        "blank_text": blank,
        "repeated_texts": int(len(non_blank) - unique_texts),
        "avg_words": avg_words,
        "missing": missing,
        "warnings": warns,
    }


# ---------- column suggestions ----------

def guess_text_column(df) -> str:
    """The column most likely to hold written comments (used when there is no 'text' column).

    Columns of dates and columns of sentiment labels are skipped. Of the rest, the one with the
    most words per value wins, unless it only has a handful of different values (a category).
    """
    best, best_score = None, -1.0
    for col in df.columns:
        series = df[col]
        if not _is_textual(series) or _looks_like_label(series):
            continue
        if _looks_like_date(col, series, check_name=False):
            continue
        sample = series.dropna().astype(str).head(500)
        if sample.empty:
            continue
        score = float(sample.str.split().str.len().mean())
        if len(sample) > 20 and sample.nunique() <= 20:
            score *= 0.2  # few different values: looks like categories, not sentences
        if score > best_score:
            best, best_score = col, score
    return best if best is not None else df.columns[0]


def _looks_like_label(series: pd.Series) -> bool:
    if not _is_textual(series):
        return False
    labels = normalize_labels(series)
    known = labels.notna()
    return bool(known.sum() >= 0.5 * len(series) and labels[known].nunique() >= 2)


def _looks_like_date(name, series: pd.Series, check_name: bool = True) -> bool:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    lowered = str(name).lower()
    if check_name and ("zone" in lowered or not any(hint in lowered for hint in DATE_NAME_HINTS)):
        return False
    sample = series.dropna().head(200)
    if sample.empty:
        return False
    return bool(parse_dates(sample).notna().mean() >= 0.8)


def suggest_columns(df, text_column) -> dict:
    """Guess which columns can be used to compare groups, explain complaints, check accuracy
    and plot sentiment over time. Each value is a column name or None."""
    others = [c for c in df.columns if c != text_column]

    label_candidates = [c for c in others if _looks_like_label(df[c])]
    label = next(
        (c for c in label_candidates if "sentiment" in str(c).lower()),
        label_candidates[0] if label_candidates else None,
    )
    reason = next(
        (
            c for c in others
            if c != label
            and "reason" in str(c).lower()
            and not any(word in str(c).lower() for word in ("gold", "confidence"))
        ),
        None,
    )
    date = next((c for c in others if c not in (label, reason) and _looks_like_date(c, df[c])), None)

    taken = {text_column, label, reason, date}
    group = next(
        (c for c in others if str(c).lower() in GROUP_NAME_HINTS and c not in taken), None
    )
    if group is None:
        for c in others:
            name = str(c).lower()
            if c in taken or not _is_textual(df[c]):
                continue
            if name.endswith("id") or "gold" in name or "confidence" in name:
                continue
            n = df[c].nunique()
            if 2 <= n <= 30 and n < 0.5 * max(len(df), 1):
                group = c
                break

    return {"group": group, "reason": reason, "label": label, "date": date}
