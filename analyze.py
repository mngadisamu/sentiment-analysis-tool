"""Command-line version of the sentiment analysis tool.

Usage:
    python analyze.py data/Tweets.csv
    python analyze.py data/Tweets.csv --report report.html
    python analyze.py data/Tweets.csv --report report.md --sections overview over_time reasons
    python analyze.py data/Tweets.csv --groups Delta United --report report.html
    python analyze.py reviews.csv --text-column review_text --group-column product --no-clean

The group, reason, label and date columns are guessed from the file. Use the --*-column
options to choose them yourself, or pass "none" to switch one off.
"""

import argparse
from pathlib import Path

import pandas as pd

from cleaning import (
    CleaningOptions,
    assess_quality,
    clean_dataframe,
    guess_text_column,
    suggest_columns,
)
from insights import sentiment_by_group, sentiment_over_time, top_reasons
from report import SECTIONS, ReportInputs, generate_report, markdown_to_html, pretty_name
from sentiment import accuracy_against, analyze_dataframe, reserved_name, sentiment_counts


def _pick(parser, df, value, suggestion, option):
    """Use the column the person named, the guess, or nothing ("none")."""
    if value is None:
        return suggestion
    if value.lower() == "none":
        return None
    if value not in df.columns:
        parser.error(f"{option}: column '{value}' was not found. Columns in the file: {', '.join(map(str, df.columns))}")
    return value


def main():
    parser = argparse.ArgumentParser(description="Classify text as positive, negative or neutral.")
    parser.add_argument("input", help="Path to a CSV file containing a text column")
    parser.add_argument("--text-column", help="Column with the text (default: 'text', or the column with the longest texts)")
    parser.add_argument("--group-column", help="Compare sentiment between the values of this column (default: guessed)")
    parser.add_argument("--reason-column", help="Column that says why a text is negative (default: guessed)")
    parser.add_argument("--label-column", help="Column with existing positive/neutral/negative labels to check against (default: guessed)")
    parser.add_argument("--date-column", help="Column with the date of each text (default: guessed)")
    parser.add_argument("--groups", "--airlines", dest="groups", nargs="+", help="Only include these values of the group column (put names with spaces in quotes)")
    parser.add_argument("--general-text", action="store_true", help="Use plain VADER instead of the customer-message word list (better for text that is not complaints or support messages)")
    parser.add_argument("--no-clean", action="store_true", help="Skip the cleaning step")
    parser.add_argument("--cleaned-output", help="Also save the cleaned data (before the sentiment columns are added) to this file")
    parser.add_argument("--output", default="results.csv", help="Where to save the results (default: results.csv)")
    parser.add_argument("--report", help="Also write a report. Use a .html or .md file name")
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=list(SECTIONS),
        metavar="SECTION",
        help="Sections to include in the report (default: all). Choices: " + ", ".join(SECTIONS),
    )
    parser.add_argument("--title", default="Sentiment Analysis Report", help="Title of the report")
    args = parser.parse_args()

    if args.report and Path(args.report).suffix.lower() not in (".html", ".md"):
        parser.error("--report must end in .html or .md")

    raw = pd.read_csv(args.input)
    text_column = args.text_column or ("text" if "text" in raw.columns else guess_text_column(raw))
    if text_column not in raw.columns:
        parser.error(f"--text-column: column '{text_column}' was not found. Columns in the file: {', '.join(map(str, raw.columns))}")

    suggested = suggest_columns(raw, text_column)
    group = _pick(parser, raw, args.group_column, suggested["group"], "--group-column")
    reason = _pick(parser, raw, args.reason_column, suggested["reason"], "--reason-column")
    label = _pick(parser, raw, args.label_column, suggested["label"], "--label-column")
    date = _pick(parser, raw, args.date_column, suggested["date"], "--date-column")

    print(f"Text column: {text_column}")
    print(f"Group: {group}  |  Reason: {reason}  |  Existing labels: {label}  |  Date: {date}\n")

    # 1. Check the file
    quality = assess_quality(raw, text_column)
    print(f"File check: {quality['rows']:,} rows, {quality['duplicate_rows']:,} duplicate rows, {quality['blank_text']:,} rows with no text")
    for warning in quality["warnings"]:
        print(f"  Warning: {warning}")
    print()

    # 2. Clean
    cleaning, data = None, raw
    if not args.no_clean:
        cleaning = clean_dataframe(raw, text_column, CleaningOptions(), date_column=date)
        data = cleaning.data
        print(f"Cleaning: {cleaning.rows_before:,} rows before, {cleaning.rows_after:,} after")
        for entry in cleaning.log:
            print(f"  {entry['step']}: {entry['affected']:,}")
        print()
        if args.cleaned_output:
            data.to_csv(args.cleaned_output, index=False)
            print(f"Cleaned data saved to {args.cleaned_output}\n")
    if data.empty:
        parser.error("No rows are left to analyze after cleaning.")

    # 3. Analyze (columns called sentiment, score or clean_text in the file are renamed original_*)
    results = analyze_dataframe(data, text_column, customer_messages=not args.general_text)
    results.to_csv(args.output, index=False)
    print(f"Analyzed {len(results):,} rows. Saved to {args.output}\n")
    group, reason, label, date = (reserved_name(c) for c in (group, reason, label, date))

    # Optional filter: applies to everything printed below and to the report
    filters_note = ""
    if args.groups:
        if group is None:
            parser.error("--groups needs a group column. Use --group-column to choose one.")
        results = results[results[group].astype(str).isin(args.groups)]
        if results.empty:
            parser.error("none of those values were found in the group column")
        filters_note = f"Only these {pretty_name(group)} values are included: " + ", ".join(args.groups) + "."
        print(f"{filters_note} ({len(results):,} rows)\n")

    print("Sentiment breakdown")
    print(sentiment_counts(results).to_string(), "\n")

    if group is not None:
        print(f"Sentiment by {pretty_name(group)} (% of texts)")
        print(sentiment_by_group(results, group).round(1).to_string(), "\n")

    if reason is not None:
        print("Top reasons among the texts the tool flagged as negative")
        print(top_reasons(results, reason, 5).to_string(), "\n")

    if date is not None:
        over_time, unit = sentiment_over_time(results, date)
        if len(over_time) >= 2:
            print(f"Sentiment over time: {len(over_time)} {unit}s (see the report for details)\n")

    if label is not None:
        acc = accuracy_against(results, label)
        print(f"Agreement with the '{pretty_name(label)}' column: {acc:.1%}\n")

    # 4. Report
    if args.report:
        inputs = ReportInputs(
            group_column=group,
            reason_column=reason,
            label_column=label,
            date_column=date,
            quality=quality,
            cleaning=cleaning,
            customer_messages=not args.general_text,
        )
        report_md = generate_report(results, args.sections, args.title, filters_note, inputs)
        if Path(args.report).suffix.lower() == ".html":
            content = markdown_to_html(report_md, args.title)
        else:
            content = report_md
        Path(args.report).write_text(content, encoding="utf-8")
        print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
