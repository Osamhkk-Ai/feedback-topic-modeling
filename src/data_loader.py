# =====================================================================
# data_loader.py  —  Load a CSV/Excel file and pick the comment column.
# =====================================================================

import os
import pandas as pd


def load_dataframe(path):
    """Load .csv / .xlsx / .xls into a DataFrame.

    CSV is read as UTF-8 (with BOM tolerance) so Arabic is preserved;
    if that fails we fall back to the legacy Windows-Arabic codepage.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            f"-> Edit INPUT_PATH in config.py, or run `python make_sample_data.py` first."
        )

    ext = os.path.splitext(path)[1].lower()

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)                       # needs openpyxl for .xlsx
    elif ext == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            print("[data_loader] UTF-8 failed, retrying with cp1256 (Windows Arabic)...")
            df = pd.read_csv(path, encoding="cp1256")
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .csv, .xlsx or .xls.")

    print(f"[data_loader] loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def pick_comment_column(df, column_name):
    """Validate that the chosen comment column exists; helpful error if not."""
    if column_name not in df.columns:
        raise KeyError(
            f"Column '{column_name}' not found.\n"
            f"-> Set COMMENT_COLUMN in config.py to one of: {list(df.columns)}"
        )
    return column_name
