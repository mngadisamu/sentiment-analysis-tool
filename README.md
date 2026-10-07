# Sentiment Analysis Tool

Upload a CSV of written comments (tweets, reviews, survey answers). The tool checks the file, cleans it, labels every comment as **positive**, **neutral** or **negative**, shows a dashboard, and writes a report you can download. It works on any CSV with a column of text, not only airline tweets.

Built for Week 3 of the CAPACITI x Clickatell AI Bootcamp (April 2026).

## What it does

1. **Checks the file**: rows, duplicates, blank texts, missing values, and warnings (for example if you picked a column that is not real text).
2. **Cleans a copy** of the data and logs every change: extra spaces, inconsistent spelling and capitals ("late flight" / "Late Flight"), blank rows, duplicate rows, optional repeated texts, and dates read as real dates. Your original file is never changed.
3. **Scores each text** with VADER, a rule-based sentiment method. Score 0.05 or more is positive, -0.05 or less is negative, anything between is neutral. **Customer messages mode** (on by default) adds a list of common complaint words and treats "help" and "please" as neutral. On the Kaggle airline data it raised agreement with the dataset's labels from 55.0% to 65.9%.
4. **Shows a dashboard**: overall breakdown, sentiment over time, sentiment by group, and top reasons for negative texts.
5. **Writes a report** (HTML or Markdown). You tick which sections to include.
6. **Downloads**: the cleaned data, the results with sentiment columns, and the report.

## Setup

Python 3.9 or newer.

```
pip install -r requirements.txt
```

## Run the dashboard

```
streamlit run app.py
```

Upload a CSV in the sidebar, or use the built-in sample. The tool guesses which columns hold the text, group, reason, existing labels and date. Change any of them in the sidebar if the guess is wrong, or choose "(none)".

## Run from the command line

```
python analyze.py data/sample_tweets.csv
python analyze.py data/sample_tweets.csv --report report.html
python analyze.py data/sample_tweets.csv --report report.md --sections overview over_time reasons
python analyze.py data/sample_tweets.csv --groups Delta United --report report.html
python analyze.py reviews.csv --text-column review_text --group-column product --date-column posted
```

| Option | What it does |
| --- | --- |
| `--text-column` | Column with the text (guessed if left out) |
| `--group-column` | Compare sentiment between the values of this column |
| `--reason-column` | Column that explains why a text is negative |
| `--label-column` | Existing positive/neutral/negative labels, used for an accuracy check |
| `--date-column` | Column with dates, used for sentiment over time |
| `--groups` | Only include these values of the group column (`--airlines` also works) |
| `--general-text` | Use plain VADER instead of customer messages mode (better for text that is not complaints or support messages) |
| `--no-clean` | Skip cleaning |
| `--cleaned-output` | Also save the cleaned data |
| `--output` | Results file (default `results.csv`) |
| `--report` | Report file, `.html` or `.md` |
| `--sections` | Report sections to include |
| `--title` | Report title |

Pass `none` to any `--*-column` option to switch it off.

Report sections: `overview`, `data_quality`, `cleaning`, `breakdown`, `over_time`, `by_group`, `reasons`, `words`, `examples`, `accuracy`, `method`. A section is left out automatically when the file has no column for it.

## Using the Kaggle airline dataset

Download `Tweets.csv` from "Twitter US Airline Sentiment" on Kaggle and upload it in the dashboard or pass it to `analyze.py`. Do not commit it to GitHub: use the small sample in `data/` instead.

`data/sample_tweets.csv` is a 30-row illustrative file, not real Kaggle data. It contains a duplicate row, a blank text and inconsistent spellings on purpose, so you can see the cleaning step work.

## Project structure

```
app.py          Streamlit dashboard
analyze.py      Command-line version
sentiment.py    Text cleaning for scoring and VADER classification
cleaning.py     Quality check, data cleaning, column suggestions
insights.py     Group, reason and over-time tables
report.py       Report writer (templated text, HTML or Markdown)
data/           Sample data
```

## Limitations

- VADER scores words, so it can miss sarcasm, context and slang. The accuracy check compares it with existing labels when the file has them.
- Customer messages mode helps on complaints but still agrees with the airline labels only about two times in three, and it still rates fewer tweets negative (49.0%) than the labels do (about 63%). Use it to compare groups and days, and treat the absolute shares with care.
- The complaint word list was written by hand, after looking at the most common words in this dataset, so the 65.9% is measured on data that informed the list. Expect a smaller gain on other data.
- It is built for English text.
- Small groups or periods give unreliable percentages. The report adds a caution when counts are low.
- The report wording comes from templates filled with the numbers, not from an AI model.
- Dates are read as written. A time-zone offset such as -0800 is ignored.
