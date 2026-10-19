"""
data_processor.py
=================
Cleaning, feature engineering and reshaping for the Sprints Talent Pipeline
project. Everything is vectorised (Pandas / NumPy ufuncs): there are no
row-wise Python loops over the learner data.

Typical use (from a notebook in /notebooks):

    import sys; sys.path.append("../src")
    import data_processor as dp

    result = dp.run_pipeline()          # reads data/raw, writes data/processed
    df = result["clean"]

All paths are relative to the project root, so the code runs anywhere the
folder tree is preserved.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("data_processor")

# --------------------------------------------------------------------------
# Paths (relative to project root = parent of /src)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_COHORT_CSV = RAW_DIR / "sprints_cohort_data_2023.csv"
RAW_REQUIREMENTS_JSON = RAW_DIR / "employer_partner_requirements.json"

# --------------------------------------------------------------------------
# Configuration -- edit here if your column names differ from the guesses
# --------------------------------------------------------------------------
SCORE_RANGE = (0, 100)          # valid range for assessment scores
PROGRAM_MONTHS = 12             # program length used to place modules on a timeline
IQR_FACTOR = 1.5                # outlier fences for engagement

# Optional: map a requirement skill name -> module column name.
# Example: {"python": "module_2", "sql": "module_5", "visualization": "module_8"}
# If left empty, skills are matched to module columns by name.
SKILL_TO_MODULE: dict[str, str] = {}

DEMOGRAPHIC_CANDIDATES = [
    "gender", "age_group", "age", "region", "country", "city",
    "education_level", "education", "background",
]


# --------------------------------------------------------------------------
# Schema detection
# --------------------------------------------------------------------------
@dataclass
class Schema:
    id_col: str | None = None
    cohort: str | None = None
    track: str | None = None
    engagement: str | None = None
    capstone: str | None = None
    modules: list[str] = field(default_factory=list)
    demographics: list[str] = field(default_factory=list)
    date: str | None = None


def snake_case(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()


def _first(cols: list[str], *keywords: str) -> str | None:
    for kw in keywords:
        for c in cols:
            if kw in c:
                return c
    return None


def detect_schema(df: pd.DataFrame) -> Schema:
    """Infer the important columns from (snake_case) column names."""
    cols = list(df.columns)
    modules = sorted(
        [c for c in cols if re.match(r"^(module|mod)_?0*\d+", c)],
        key=lambda c: int(re.search(r"(\d+)", c).group(1)),
    )
    s = Schema(
        id_col=next((c for c in cols if re.search(r"(^|_)id$", c)), None),
        cohort=_first(cols, "cohort"),
        track=_first(cols, "track", "program", "pathway"),
        engagement=_first(cols, "engagement"),
        capstone=_first(cols, "capstone", "final"),
        modules=modules,
        demographics=[c for c in DEMOGRAPHIC_CANDIDATES if c in cols],
        date=_first(cols, "date", "timestamp", "enroll"),
    )
    if len(s.modules) != 10:
        log.warning("Expected 10 module columns, found %d: %s", len(s.modules), s.modules)
    return s


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw(path: Path | str = RAW_COHORT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [snake_case(c) for c in df.columns]
    log.info("Loaded %s rows x %s columns", *df.shape)
    return df


def load_requirements(path: Path | str = RAW_REQUIREMENTS_JSON) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame, schema: Schema | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Returns (clean_df, report). `report` records every cleaning decision so it
    can be quoted in the notebook and the PDF report.
    """
    schema = schema or detect_schema(df)
    out = df.copy()
    report: dict = {"rows_raw": len(out)}

    # 1. Tidy text columns and coerce numeric ones
    text_cols = out.select_dtypes(include="object").columns
    for c in text_cols:
        out[c] = out[c].astype("string").str.strip()
    numeric_cols = [c for c in schema.modules + [schema.engagement, schema.capstone] if c]
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # 2. Duplicates
    subset = [schema.id_col] if schema.id_col else None
    n_dup = int(out.duplicated(subset=subset).sum())
    out = out.drop_duplicates(subset=subset, keep="first")
    report["duplicates_removed"] = n_dup

    # 3. Impossible scores -> NaN (then imputed below)
    lo, hi = SCORE_RANGE
    score_cols = schema.modules + ([schema.capstone] if schema.capstone else [])
    out_of_range = ((out[score_cols] < lo) | (out[score_cols] > hi)).sum().sum()
    out[score_cols] = out[score_cols].where(out[score_cols].between(lo, hi))
    report["out_of_range_scores_nulled"] = int(out_of_range)

    # 4. Missing values
    report["missing_before"] = out.isna().sum()[lambda s: s > 0].to_dict()
    all_modules_missing = out[schema.modules].isna().all(axis=1)
    report["rows_dropped_no_scores"] = int(all_modules_missing.sum())
    out = out.loc[~all_modules_missing]

    group_key = schema.cohort if schema.cohort else None
    for c in numeric_cols:
        if group_key:
            med = out.groupby(group_key)[c].transform("median")
            out[c] = out[c].fillna(med)
        out[c] = out[c].fillna(out[c].median())
    cat_cols = [c for c in out.columns if c not in numeric_cols and out[c].dtype.name in ("string", "object")]
    out[cat_cols] = out[cat_cols].fillna("Unknown")
    report["imputation"] = "numeric: cohort median (fallback global median); categorical: 'Unknown'"

    # 5. Engagement outliers: IQR winsorising (keeps rows, caps extremes)
    if schema.engagement:
        e = out[schema.engagement]
        q1, q3 = e.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr
        flag = (e < lower) | (e > upper)
        out["engagement_outlier"] = flag
        out[schema.engagement] = e.clip(lower, upper)
        report["engagement_outliers_capped"] = int(flag.sum())
        report["engagement_fences"] = (float(lower), float(upper))

    out = out.reset_index(drop=True)
    report["rows_clean"] = len(out)
    log.info("Cleaning done: %s -> %s rows", report["rows_raw"], report["rows_clean"])
    return out, report


# --------------------------------------------------------------------------
# Feature engineering (NumPy ufuncs, no loops)
# --------------------------------------------------------------------------
def default_weights(n_modules: int, start: float = 1.0, end: float = 2.0) -> np.ndarray:
    """Later modules weigh more (they are more advanced). Normalised to sum 1."""
    w = np.linspace(start, end, n_modules)
    return np.divide(w, w.sum())


def add_skill_mastery_index(
    df: pd.DataFrame, schema: Schema, weights: np.ndarray | None = None
) -> pd.DataFrame:
    """Weighted average across the module columns, NaN-safe, fully vectorised."""
    out = df.copy()
    vals = out[schema.modules].to_numpy(dtype=float)
    w = default_weights(len(schema.modules)) if weights is None else np.asarray(weights, float)
    present = np.logical_not(np.isnan(vals))
    numerator = np.nansum(np.multiply(vals, w), axis=1)
    denominator = np.sum(np.multiply(present, w), axis=1)
    out["skill_mastery_index"] = np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
    )
    return out


# --------------------------------------------------------------------------
# Employer requirements -> matrix -> compatibility (NumPy linalg)
# --------------------------------------------------------------------------
_NAME_KEYS = ("employer", "company", "name", "partner")
_SKILL_KEYS = ("required_skills", "technical_requirements", "skills", "requirements")


def _iter_employers(req: dict | list):
    """Yield (employer_name, {skill: weight}) from a few plausible JSON shapes."""
    items = req
    if isinstance(req, dict):
        for k in ("employers", "partners", "companies", "requisitions", "jobs"):
            if k in req and isinstance(req[k], list):
                items = req[k]
                break
        else:
            # {"Acme": {...skills...}} shape
            items = [{"name": k, **v} if isinstance(v, dict) else {"name": k, "skills": v}
                     for k, v in req.items()]
    for i, item in enumerate(items):
        name = next((str(item[k]) for k in _NAME_KEYS if k in item), f"employer_{i + 1}")
        raw = next((item[k] for k in _SKILL_KEYS if k in item), {})
        if isinstance(raw, list):
            raw = {(s["skill"] if isinstance(s, dict) else s): (s.get("weight", 1) if isinstance(s, dict) else 1)
                   for s in raw}
        yield name, {snake_case(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def build_requirement_matrix(
    requirements: dict | list, module_cols: list[str], skill_to_module: dict[str, str] | None = None
) -> tuple[list[str], np.ndarray, list[str]]:
    """
    Returns (employer_names, R, unmatched_skills) where R has shape
    (n_employers, n_modules). Skills are mapped to module columns through
    `skill_to_module` or by substring match on the module column name.
    """
    mapping = {snake_case(k): v for k, v in (skill_to_module or SKILL_TO_MODULE).items()}
    col_index = {c: j for j, c in enumerate(module_cols)}
    employers = list(_iter_employers(requirements))
    R = np.zeros((len(employers), len(module_cols)))
    unmatched: list[str] = []
    for i, (_, skills) in enumerate(employers):
        for skill, weight in skills.items():
            target = mapping.get(skill) or next((c for c in module_cols if skill in c), None)
            if target in col_index:
                R[i, col_index[target]] += weight
            else:
                unmatched.append(skill)
    if unmatched:
        log.warning("Skills with no module mapping (set SKILL_TO_MODULE): %s", sorted(set(unmatched)))
    return [n for n, _ in employers], R, sorted(set(unmatched))


def compatibility_scores(profiles: np.ndarray, requirements: np.ndarray) -> np.ndarray:
    """Cosine similarity via one matrix product: (n_profiles x n_employers)."""
    num = profiles @ requirements.T
    p_norm = np.linalg.norm(profiles, axis=1, keepdims=True)
    r_norm = np.linalg.norm(requirements, axis=1, keepdims=True).T
    den = p_norm @ r_norm
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def add_compatibility(
    df: pd.DataFrame, schema: Schema, requirements: dict | list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Learner-level compatibility columns (compat_<employer>, best_match_employer)
    and a cohort-level matrix of aggregated profiles vs employers.
    """
    names, R, _ = build_requirement_matrix(requirements, schema.modules)
    L = df[schema.modules].to_numpy(dtype=float)
    scores = compatibility_scores(L, R)
    compat = pd.DataFrame(scores, columns=[f"compat_{snake_case(n)}" for n in names], index=df.index)
    out = pd.concat([df, compat], axis=1)
    out["best_match_employer"] = np.array(names)[scores.argmax(axis=1)]
    out["best_match_score"] = scores.max(axis=1)

    group = schema.cohort or schema.track
    if group:
        prof = df.groupby(group)[schema.modules].mean()
        cohort_scores = compatibility_scores(prof.to_numpy(float), R)
        cohort_matrix = pd.DataFrame(cohort_scores, index=prof.index, columns=names)
    else:
        cohort_matrix = pd.DataFrame()
    return out, cohort_matrix


# --------------------------------------------------------------------------
# Reshaping and summaries
# --------------------------------------------------------------------------
def to_long(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """
    Wide -> long view of assessment progression (one row per learner x module).
    `program_month` spreads the modules evenly over PROGRAM_MONTHS because the
    dataset has no per-module dates (documented assumption).
    """
    id_vars = [c for c in [schema.id_col, schema.cohort, schema.track, *schema.demographics,
                           schema.engagement, "skill_mastery_index"] if c and c in df.columns]
    if not schema.id_col:
        df = df.assign(learner_idx=np.arange(len(df)))
        id_vars = ["learner_idx", *id_vars]
    long = df.melt(id_vars=id_vars, value_vars=schema.modules,
                   var_name="module", value_name="score")
    long["module_number"] = long["module"].str.extract(r"(\d+)").astype(int)
    n = len(schema.modules)
    long["program_month"] = np.ceil(long["module_number"] * PROGRAM_MONTHS / n).astype(int)
    return long


def summarize(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Median SMI and engagement by cohort and each demographic grouping."""
    keys = [c for c in [schema.cohort, *schema.demographics] if c]
    named = {"median_smi": ("skill_mastery_index", "median"), "learners": ("skill_mastery_index", "size")}
    if schema.engagement:
        named["median_engagement"] = (schema.engagement, "median")
    return df.groupby(keys, observed=True).agg(**named).reset_index().sort_values("median_smi", ascending=False)


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------
def run_pipeline(save: bool = True) -> dict:
    raw = load_raw()
    schema = detect_schema(raw)
    clean_df, report = clean(raw, schema)
    feat = add_skill_mastery_index(clean_df, schema)

    requirements = load_requirements()
    feat, cohort_compat = add_compatibility(feat, schema, requirements)

    long = to_long(feat, schema)
    summary = summarize(feat, schema)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        feat.to_csv(PROCESSED_DIR / "cohort_clean_features.csv", index=False)
        long.to_csv(PROCESSED_DIR / "cohort_long.csv", index=False)
        summary.to_csv(PROCESSED_DIR / "cohort_summary.csv", index=False)
        cohort_compat.to_csv(PROCESSED_DIR / "cohort_employer_compatibility.csv")
        log.info("Saved processed files to %s", PROCESSED_DIR)

    return {"schema": schema, "clean": feat, "long": long, "summary": summary,
            "cohort_compat": cohort_compat, "report": report}


if __name__ == "__main__":
    res = run_pipeline()
    print(json.dumps(res["report"], indent=2, default=str))