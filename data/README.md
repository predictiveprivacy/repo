# Data

This directory contains the **survey response data** used throughout the analysis pipeline, along with utilities for anonymizing sensitive identifiers. All datasets stored here are safe for analysis and sharing within the constraints described below.

**Important:** All Prolific participant identifiers have been anonymized prior to analysis using the provided `anonymize.py` script.

---

## Directory Structure

```text
data/
├── anonymize.py
├── survey1_results.csv
├── survey2_results.csv
└── README.md
```

---

## Anonymization & Privacy

### Prolific ID Handling

- Raw Prolific IDs are **not stored** in this repository
- Before any analysis, participant identifiers were anonymized using `anonymize.py`
- Anonymization replaces Prolific IDs with non-identifying surrogate IDs

This ensures:
- No participant can be re-identified
- Data remains internally consistent for analysis
- Compliance with ethical research and IRB-style standards

---

### `anonymize.py`
**Purpose**
- Remove or transform personally identifying information
- Produce analysis-safe datasets from raw Prolific exports

**Key Responsibilities**
- Load raw Prolific survey CSVs
- Replace Prolific IDs with anonymized identifiers
- Preserve survey structure and row integrity
- Export sanitized CSV files for downstream use

**Usage**
```bash
python anonymize.py --input "/path/to/raw_survey_results.csv"
```

> Raw input files are intentionally excluded from version control.  
> The script outputs an anonymized version of the input file for downstream analysis.

---

## Survey Data Files

### `survey1_results.csv`
- Anonymized responses from **Survey 1**
- Used as input for statistical analysis and visualization

---

### `survey2_results.csv`
- Anonymized responses from **Survey 2**
- Includes multiple certainty conditions used for comparative analysis

---

## Data Integrity Notes

- No manual edits after anonymization
- All cleaning and exclusions are scripted
- Survey datasets are kept separate by design
- Column schemas are stable across runs
