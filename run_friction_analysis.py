"""
Run the administrative friction analysis end-to-end.

Defaults assume this file is located at waymark-local/packaging/admin_friction
and data are in ../../data/real_inputs. Outputs are written to
../../notebooks/admin_friction/outputs.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

# Use non-interactive backend for CLI environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from lifelines.statistics import logrank_test, proportional_hazard_test  # noqa: E402
from lifelines.plotting import add_at_risk_counts  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import roc_curve, auc  # noqa: E402
import joblib  # noqa: E402


def get_patterns():
    return {
        "scheduling": [
            r"appointment",
            r"\bappt\b",
            r"schedule",
            r"scheduling",
            r"resched",
            r"availability",
            r"opening",
            r"booked",
            r"wait\s?list",
            r"on hold",
            r"busy signal",
            r"hold time",
            r"call back later",
            r"office closed",
            r"not accepting",
            r"next available",
            r"new patient",
            r"pcp",
            r"provider",
            r"doctor",
            r"clinic",
        ],
        "transportation": [
            r"\bride\b",
            r"transport",
            r"bus",
            r"lyft",
            r"uber",
            r"cab",
            r"taxi",
            r"shuttle",
            r"paratransit",
            r"modiv",
            r"vta",
            r"drive",
            r"car trouble",
            r"no car",
            r"gas card",
            r"bus pass",
            r"need.*ride",
            r"van",
            r"wheelchair van",
        ],
        "documentation": [
            r"paperwork",
            r"forms?",
            r"application",
            r"recert",
            r"renewal",
            r"fax",
            r"document",
            r"proof of",
            r"verification",
            r"id card",
            r"signature",
            r"mailed",
            r"mailing",
            r"upload",
            r"submit",
            r"submitted",
            r"letter",
            r"notice",
        ],
        "authorization": [
            r"prior auth",
            r"authorization",
            r"pre[- ]?auth",
            r"approval",
            r"denied",
            r"denial",
            r"not covered",
            r"pending",
            r"appeal",
            r"pharmacy refusal",
            r"needs auth",
            r"insurance will not",
            r"pa request",
        ],
    }


def flag_notes(notes: pd.DataFrame) -> pd.DataFrame:
    """Apply regex patterns to notes; add friction flags (baseline rules)."""
    patterns = {k: re.compile("|".join(v)) for k, v in get_patterns().items()}
    exclude = {
        "scheduling": re.compile(r"pharmacy outreach|\badh\b|medication|meds|\brx\b|prescription")
    }
    notes["text_proc"] = notes["text"].fillna("").str.lower()
    for cat, regex in patterns.items():
        flag = notes["text_proc"].str.contains(regex, regex=True, na=False)
        if cat in exclude:
            flag = flag & ~notes["text_proc"].str.contains(exclude[cat], regex=True, na=False)
        notes[f"{cat}_flag"] = flag
    flag_cols = [c for c in notes.columns if c.endswith("_flag")]
    notes["friction_any"] = notes[flag_cols].any(axis=1)
    return notes


def flag_conditions(notes: pd.DataFrame) -> pd.DataFrame:
    """Heuristic condition flags based on note text."""
    if "text_proc" not in notes.columns:
        notes["text_proc"] = notes["text"].fillna("").str.lower()
    cond_patterns = {
        "cond_diabetes": r"diabet",
        "cond_hypertension": r"hypertension|htn|high blood pressure",
        "cond_copd_asthma": r"copd|asthma|bronchitis|emphysema",
        "cond_heart_failure": r"heart failure|hfref|hfpef|chf|congestive heart",
        "cond_ckd": r"ckd|chronic kidney|renal failure",
        "cond_depression_anxiety": r"depression|depressive|anxiety|panic disorder|gad\b",
        "cond_substance_use": r"alcohol use|etoh|substance|opioid|heroin|cocaine|meth|amphetamine|marijuana|cannabis",
    }
    for col, pat in cond_patterns.items():
        notes[col] = notes["text_proc"].str.contains(pat, regex=True, na=False)
    return notes


def load_core_data(data_dir: Path):
    notes = pd.read_csv(
        data_dir / "notes" / "encounter notes.csv",
        usecols=["Encounter ID", "WaymarkId", "dateOfEncounter", "text"],
    )
    notes["note_date"] = pd.to_datetime(notes["dateOfEncounter"], errors="coerce").dt.tz_localize(None)

    elig = pd.read_parquet(
        data_dir / "eligibility.parquet",
        columns=["person_id", "member_id", "zip_code", "enrollment_start_date", "enrollment_end_date"],
    ).drop_duplicates()
    mpm = pd.read_parquet(
        data_dir / "member_patient_map.parquet",
        columns=["member_id", "patient_id"],
    ).drop_duplicates()
    attr = pd.read_parquet(
        data_dir / "member_attributes.parquet",
        columns=["member_id", "gender", "birth_date", "race", "ethnicity", "risk_score", "state"],
    )
    hosp = pd.read_parquet(
        data_dir / "hospital_visits.parquet",
        columns=["patient_id", "admit_date"],
    )
    outcomes = pd.read_parquet(data_dir / "outcomes_monthly.parquet")
    return notes, elig, mpm, attr, hosp, outcomes


def build_linked_notes(notes, elig, mpm, attr):
    person_member = elig[["person_id", "member_id"]].drop_duplicates()
    person_member_patient = person_member.merge(mpm, on="member_id", how="left")
    linked = (
        notes.merge(person_member_patient, left_on="WaymarkId", right_on="person_id", how="left")
        .merge(attr.rename(columns={"state": "attr_state"}), on="member_id", how="left")
    )
    linked = linked.dropna(subset=["patient_id"]).copy()
    linked["birth_date"] = pd.to_datetime(linked["birth_date"], errors="coerce")
    linked["age"] = ((pd.Timestamp("2025-01-01") - linked["birth_date"]).dt.days / 365.25).clip(lower=0, upper=110)
    return linked


def compute_time_cost(notes):
    minutes = {
        "scheduling_flag": 8.1,
        "transportation_flag": 37.5,
        "documentation_flag": 23,
        "authorization_flag": 20,
    }
    wage_per_hour = 7.25  # federal minimum wage (lower bound)
    wage_per_hour_upper = 22.0  # MIT Living Wage, state-weighted (central estimate)
    wage_per_hour_rbrvs = 33.40  # RBRVS conversion factor (upper bound)
    for col in minutes:
        if col not in notes:
            notes[col] = False
    notes["friction_minutes"] = sum(notes[col] * mins for col, mins in minutes.items())
    notes["time_cost"] = (notes["friction_minutes"] / 60.0) * wage_per_hour
    notes["time_cost_upper"] = (notes["friction_minutes"] / 60.0) * wage_per_hour_upper
    notes["time_cost_rbrvs"] = (notes["friction_minutes"] / 60.0) * wage_per_hour_rbrvs
    return notes


def train_ml_classifiers(annot_path: Path, model_dir: Path, threshold: float = 0.5):
    """Train one-vs-rest logistic regression classifiers using adjudicated labels."""
    # Barrier-specific thresholds (transportation raised to prioritize precision
    # given low base rate; paperwork lowered to prioritize recall)
    BARRIER_THRESHOLDS = {
        "scheduling": 0.50,
        "transportation": 0.60,
        "documentation": 0.45,  # paperwork
        "authorization": 0.50,
    }
    annot = pd.read_csv(annot_path)
    cats = ["scheduling", "transportation", "documentation", "authorization"]
    y = {cat: annot[f"{cat}_gold"] for cat in cats}
    X_text = annot["text"].fillna("")

    # Temporal split for quasi-prospective validation
    dates = pd.to_datetime(annot.get("dateOfEncounter"))
    if dates.notna().any():
        q1, q2 = dates.quantile([0.6, 0.8])
        split = pd.Series("train", index=annot.index)
        split.loc[(dates > q1) & (dates <= q2)] = "val"
        split.loc[dates > q2] = "test"
    else:
        rng = np.random.default_rng(13)
        probs = rng.random(len(annot))
        split = pd.Series(np.where(probs < 0.6, "train", np.where(probs < 0.8, "val", "test")), index=annot.index)

    vectorizer = TfidfVectorizer(
        min_df=2, max_df=0.90, ngram_range=(1, 2), max_features=5000,
        sublinear_tf=True, norm="l2",
    )
    X = vectorizer.fit_transform(X_text)

    models = {}
    metrics_holdout = {}
    metrics_val = {}
    metrics_test = {}
    roc_data = {}
    for cat in cats:
        y_cat = y[cat].dropna().astype(int)
        X_cat = X[y[cat].notna()]
        split_cat = split[y[cat].notna()]
        train_mask = split_cat == "train"
        val_mask = split_cat == "val"
        test_mask = split_cat == "test"
        X_train, y_train = X_cat[train_mask], y_cat[train_mask]
        X_val, y_val = X_cat[val_mask], y_cat[val_mask]
        X_test, y_test = X_cat[test_mask], y_cat[test_mask]
        # Fallback to stratified random if any split too small
        if min(len(y_train), len(y_val), len(y_test)) < 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X_cat, y_cat, test_size=0.2, random_state=13, stratify=y_cat
            )
            X_val, y_val = X_test, y_test
        best_c, best_f1 = 1.0, -1.0
        for c_val in [0.01, 0.1, 1.0, 10.0]:
            cv_clf = LogisticRegression(
                penalty="l2", C=c_val, max_iter=1000,
                class_weight="balanced", solver="liblinear",
            )
            cv_clf.fit(X_train, y_train)
            proba_v = cv_clf.predict_proba(X_val)[:, 1]
            y_v = (proba_v >= BARRIER_THRESHOLDS[cat]).astype(int)
            m_v = prf(pd.Series(y_v), pd.Series(y_val))
            if m_v["f1"] == m_v["f1"] and m_v["f1"] > best_f1:
                best_f1, best_c = m_v["f1"], c_val
        clf = LogisticRegression(
            penalty="l2", C=best_c, max_iter=1000,
            class_weight="balanced", solver="liblinear",
        )
        clf.fit(X_train, y_train)
        models[cat] = clf
        # Test split metrics
        proba_test = clf.predict_proba(X_test)[:, 1]
        y_pred_test = (proba_test >= BARRIER_THRESHOLDS[cat]).astype(int)
        m_test = prf(pd.Series(y_pred_test), pd.Series(y_test))
        m_test.update(prf_ci(y_pred_test, y_test))
        metrics_test[cat] = m_test
        fpr, tpr, _ = roc_curve(y_test, proba_test)
        roc_data[cat] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}
        # Validation split metrics
        proba_val = clf.predict_proba(X_val)[:, 1]
        y_pred_val = (proba_val >= BARRIER_THRESHOLDS[cat]).astype(int)
        metrics_val[cat] = prf(pd.Series(y_pred_val), pd.Series(y_val))
        # Random holdout (legacy) for comparability
        metrics_holdout[cat] = metrics_test[cat]

    # Persist models and vectorizer
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "tfidf.joblib")
    for cat, clf in models.items():
        joblib.dump(clf, model_dir / f"{cat}_clf.joblib")

    metrics = {"test": metrics_test, "val": metrics_val, "holdout": metrics_holdout}
    return vectorizer, models, metrics, roc_data, BARRIER_THRESHOLDS


def apply_ml(models, vectorizer, notes: pd.DataFrame, thresholds: dict = None) -> pd.DataFrame:
    default_thresholds = {
        "scheduling": 0.50, "transportation": 0.60,
        "documentation": 0.45, "authorization": 0.50,
    }
    thresholds = thresholds or default_thresholds
    texts = notes["text"].fillna("")
    X_all = vectorizer.transform(texts)
    for cat, clf in models.items():
        proba = clf.predict_proba(X_all)[:, 1]
        notes[f"{cat}_flag"] = proba >= thresholds.get(cat, 0.50)
        notes[f"{cat}_prob"] = proba
    flag_cols = [c for c in notes.columns if c.endswith("_flag")]
    notes["friction_any"] = notes[flag_cols].any(axis=1)
    return notes


def estimate_acute_increment(outcomes: pd.DataFrame) -> float:
    outcomes = outcomes.copy()
    outcomes["medical_paid"] = outcomes["medical_paid"].fillna(outcomes["total_paid"]).fillna(0)
    base_mask = (
        (outcomes["emergency_department_ct"].fillna(0) == 0)
        & (outcomes["acute_inpatient_ct"].fillna(0) == 0)
        & (outcomes["outpatient_hospital_or_clinic_ct"].fillna(0) == 0)
        & (outcomes["telehealth_ct"].fillna(0) == 0)
    )
    baseline_med = outcomes.loc[base_mask, "medical_paid"]
    baseline_med = baseline_med.median() if len(baseline_med) > 0 else 0
    outcomes["acute_events"] = outcomes["emergency_department_ct"].fillna(0) + outcomes["acute_inpatient_ct"].fillna(0)
    acute_months = outcomes[outcomes["acute_events"] > 0].copy()
    if len(acute_months) == 0 or acute_months["acute_events"].sum() == 0:
        return 0
    acute_increment = ((acute_months["medical_paid"] - baseline_med).clip(lower=0).sum()) / acute_months[
        "acute_events"
    ].sum()
    return float(acute_increment)


def match_barriers_to_acute(barriers: pd.DataFrame, hosp: pd.DataFrame, per_event_escalation: float):
    hosp = hosp.copy()
    hosp["admit_dt"] = pd.to_datetime(hosp["admit_date"], errors="coerce")
    if str(hosp["admit_dt"].dtype).startswith("datetime64[ns,"):
        hosp["admit_dt"] = hosp["admit_dt"].dt.tz_localize(None)
    hosp = hosp.dropna(subset=["patient_id", "admit_dt"])
    bar_map = barriers.groupby("patient_id")["note_dt"].apply(list).to_dict()
    ac_events = 0
    for pid, group in hosp.groupby("patient_id"):
        bars = bar_map.get(pid)
        if not bars:
            continue
        for admit in group["admit_dt"]:
            if any((admit - b).days >= 0 and (admit - b).days <= 14 for b in bars):
                ac_events += 1
    aec_cost = ac_events * per_event_escalation
    return ac_events, aec_cost


def survival_analysis(notes_valid: pd.DataFrame, hosp: pd.DataFrame, out_path: Path, landmark_days: int = 180, outcome: str = "ed"):
    """Landmark analysis to mitigate immortal time bias."""
    hosp = hosp.copy()
    hosp["admit_dt"] = pd.to_datetime(hosp["admit_date"], errors="coerce")
    if str(hosp["admit_dt"].dtype).startswith("datetime64[ns,"):
        hosp["admit_dt"] = hosp["admit_dt"].dt.tz_localize(None)
    hosp = hosp.dropna(subset=["patient_id", "admit_dt"])
    # Filter outcome type
    cls = hosp.get("patient_class_code")
    if cls is not None:
        hosp["patient_class_code"] = cls
    def is_event(row):
        code = row.get("patient_class_code", "")
        if outcome == "ed":
            return code == "E"
        if outcome == "inpatient":
            return code == "I"
        return True  # combined acute
    hosp = hosp[hosp.apply(is_event, axis=1)].copy()

    note_grouped = notes_valid.groupby("patient_id")
    note_dates = note_grouped["note_date"].agg(["min", "max"]).rename(columns={"min": "first_note", "max": "last_note"})
    encounter_counts = note_grouped.size().rename("encounter_count")
    sched_dates = notes_valid[notes_valid["scheduling_flag"]].groupby("patient_id")["note_date"].min().rename(
        "sched_date"
    )
    patient_df = note_dates.join(sched_dates, how="left").join(encounter_counts, how="left")
    patient_df["landmark"] = patient_df["first_note"] + pd.Timedelta(days=landmark_days)

    # Define exposure if scheduling barrier before landmark
    patient_df["sched_barrier"] = (patient_df["sched_date"].notna()) & (patient_df["sched_date"] <= patient_df["landmark"])

    # Event: first ED after landmark
    ed_dates = (
        hosp.merge(patient_df[["landmark"]].reset_index(), on="patient_id", how="inner")
        .loc[lambda d: d["admit_dt"] >= d["landmark"]]
        .groupby("patient_id")["admit_dt"]
        .min()
        .rename("first_ed_post_landmark")
    )
    patient_df = patient_df.join(ed_dates, how="left")
    patient_df["event"] = patient_df["first_ed_post_landmark"].notna()
    patient_df["follow_end"] = patient_df[["last_note", "first_ed_post_landmark"]].max(axis=1)

    # Exclude if landmark after follow_end (insufficient follow-up)
    patient_df = patient_df.loc[patient_df["follow_end"] >= patient_df["landmark"]].copy()
    patient_df["time"] = (
        (patient_df["first_ed_post_landmark"].fillna(patient_df["follow_end"])) - patient_df["landmark"]
    ).dt.days.clip(lower=0)

    # Covariates
    patient_df = patient_df.reset_index().rename(columns={"index": "patient_id"}).set_index("patient_id")
    covars = notes_valid[["patient_id", "age", "gender", "ethnicity"]].drop_duplicates("patient_id").set_index("patient_id")
    patient_df = patient_df.join(covars, how="left")
    patient_df["gender"] = patient_df["gender"].fillna("Unknown")
    patient_df["ethnicity"] = patient_df["ethnicity"].fillna("Unknown")
    patient_df["age"] = patient_df["age"].fillna(patient_df["age"].median())

    # Prior ED before landmark
    prior_counts = (
        hosp.merge(patient_df[["landmark"]].reset_index(), on="patient_id", how="inner")
        .loc[lambda d: d["admit_dt"] < d["landmark"]]
        .groupby("patient_id")
        .size()
    )
    patient_df["prior_ed"] = patient_df.index.map(prior_counts).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    km_fitters = []
    for flag, label in [(False, "No barrier ≤ landmark"), (True, "Scheduling barrier ≤ landmark")]:
        mask = patient_df["sched_barrier"] == flag
        if mask.sum() == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(patient_df.loc[mask, "time"], event_observed=patient_df.loc[mask, "event"], label=label)
        kmf.plot_survival_function(ax=ax)
        km_fitters.append(kmf)
    # Cap x-axis at study duration (≈1100 days)
    max_study_days = min(1100, patient_df["time"].max())
    ax.set_xlim(0, max_study_days)
    if km_fitters:
        add_at_risk_counts(*km_fitters, ax=ax)
    ax.set_xlabel("Days from landmark")
    ax.set_ylabel("Event-free survival probability")
    ax.set_title("Kaplan-Meier: Scheduling Friction (Landmark)")
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    # Add inset zoomed to first 365 days
    ax_inset = ax.inset_axes([0.50, 0.45, 0.45, 0.45])
    for flag, label in [(False, "No barrier ≤ landmark"), (True, "Scheduling barrier ≤ landmark")]:
        mask = patient_df["sched_barrier"] == flag
        if mask.sum() == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(patient_df.loc[mask, "time"], event_observed=patient_df.loc[mask, "event"], label=label)
        kmf.plot_survival_function(ax=ax_inset, ci_show=False, legend=False)
    ax_inset.set_xlim(0, 365)
    ax_inset.set_ylim(0, 1.0)
    ax_inset.set_xlabel("Days (0–365)", fontsize=8)
    ax_inset.set_ylabel("", fontsize=8)
    ax_inset.tick_params(labelsize=7)
    ax_inset.set_title("First year detail", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    # Truncated view (0-180 days post-landmark)
    fig, ax = plt.subplots(figsize=(7, 5))
    for flag, label in [(False, "No barrier ≤ landmark"), (True, "Scheduling barrier ≤ landmark")]:
        mask = patient_df["sched_barrier"] == flag
        if mask.sum() == 0:
            continue
        kmf_t = KaplanMeierFitter()
        kmf_t.fit(patient_df.loc[mask, "time"], event_observed=patient_df.loc[mask, "event"], label=label)
        kmf_t.plot_survival_function(ax=ax)
    ax.set_xlim(0, 180)
    ax.set_xlabel("Days from landmark")
    ax.set_ylabel("Event-free survival probability")
    ax.set_title("Kaplan-Meier 0–180 Days Post-Landmark")
    fig.tight_layout()
    fig.savefig(out_path.with_name(out_path.stem + "_180.png"), dpi=300)
    plt.close(fig)

    # Log-rank test between exposed vs unexposed
    lr_p = None
    try:
        mask_exp = patient_df["sched_barrier"] == True  # noqa: E712
        mask_unexp = patient_df["sched_barrier"] == False  # noqa: E712
        if mask_exp.sum() > 0 and mask_unexp.sum() > 0:
            lr_res = logrank_test(
                patient_df.loc[mask_unexp, "time"],
                patient_df.loc[mask_exp, "time"],
                event_observed_A=patient_df.loc[mask_unexp, "event"],
                event_observed_B=patient_df.loc[mask_exp, "event"],
            )
            lr_p = float(lr_res.p_value)
    except Exception:
        lr_p = None

    cph = CoxPHFitter(penalizer=0.1)
    cov_df = patient_df[["time", "event", "sched_barrier", "age", "prior_ed", "encounter_count"]].copy()
    cov_df = cov_df.join(pd.get_dummies(patient_df["gender"], prefix="gender"))
    cov_df = cov_df.join(pd.get_dummies(patient_df["ethnicity"], prefix="eth"))
    cov_df_clean = cov_df.dropna()
    # Drop near-constant covariates to stabilize PH model
    var_mask = cov_df_clean.drop(columns=["time", "event"]).var() > 1e-8
    cols_keep = ["time", "event"] + [c for c in cov_df_clean.columns if c not in ["time", "event"] and var_mask.get(c, False)]
    cov_df_clean = cov_df_clean[cols_keep]
    hr_sched = None
    ci = None
    ph_p = None
    cph_summary = None
    if len(cov_df_clean) > 0:
        try:
            cph.fit(cov_df_clean, duration_col="time", event_col="event")
            cph_summary = cph.summary.reset_index()
            if "sched_barrier" in cph.params_:
                hr_sched = float(np.exp(cph.params_["sched_barrier"]))
                ci_raw = cph.confidence_intervals_.loc["sched_barrier"]
                ci = list(np.exp(ci_raw.values))
            try:
                ph_test = proportional_hazard_test(cph, cov_df_clean, time_transform="rank")
                if "sched_barrier" in ph_test.summary.index:
                    ph_p = float(ph_test.summary.loc["sched_barrier", "p"])
            except Exception:
                ph_p = None
        except Exception:
            hr_sched = None
            ci = None
            cph_summary = None
            ph_p = None
    return hr_sched, ci, patient_df, cph_summary, lr_p, ph_p


def make_annotation_sample(notes_valid: pd.DataFrame, out_path: Path):
    notes = notes_valid.copy()
    flag_cols = [c for c in notes.columns if c.endswith("_flag")]
    notes["flag_sum"] = notes[flag_cols].sum(axis=1)
    strata = {
        "scheduling_only": (notes["scheduling_flag"]) & (notes["flag_sum"] == 1),
        "transportation_only": (notes["transportation_flag"]) & (notes["flag_sum"] == 1),
        "documentation_only": (notes["documentation_flag"]) & (notes["flag_sum"] == 1),
        "authorization_only": (notes["authorization_flag"]) & (notes["flag_sum"] == 1),
        "multi_barrier": notes["flag_sum"] >= 2,
        "no_barrier": notes["flag_sum"] == 0,
    }
    sizes = {
        "scheduling_only": 300,
        "transportation_only": 300,
        "documentation_only": 300,
        "authorization_only": 300,
        "multi_barrier": 300,
        "no_barrier": 500,
    }
    frames = []
    for name, mask in strata.items():
        subset = notes[mask]
        n = sizes.get(name, 0)
        sample = subset if len(subset) <= n else subset.sample(n, random_state=13)
        sample = sample.assign(stratum=name)
        frames.append(sample)
    sampled = pd.concat(frames).sample(frac=1, random_state=13).reset_index(drop=True)
    annot = sampled[
        [
            "Encounter ID",
            "WaymarkId",
            "patient_id",
            "dateOfEncounter",
            "text",
            "scheduling_flag",
            "transportation_flag",
            "documentation_flag",
            "authorization_flag",
            "stratum",
        ]
    ]
    annot.to_csv(out_path, index=False)
    return len(annot)


def prf(pred, gold):
    pred_arr = np.asarray(pred).astype(int)
    gold_arr = np.asarray(gold).astype(int)
    tp = int(((pred_arr == 1) & (gold_arr == 1)).sum())
    fp = int(((pred_arr == 1) & (gold_arr == 0)).sum())
    fn = int(((pred_arr == 0) & (gold_arr == 1)).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else float("nan")
    rec = tp / (tp + fn) if tp + fn > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) > 0 else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def prf_ci(pred, gold, n_boot: int = 1000, seed: int = 13):
    """Bootstrap CIs for precision/recall/F1."""
    pred_arr = np.asarray(pred).astype(int)
    gold_arr = np.asarray(gold).astype(int)
    n = len(pred_arr)
    rng = np.random.default_rng(seed)
    precs, recs, f1s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = prf(pred_arr[idx], gold_arr[idx])
        precs.append(m["precision"])
        recs.append(m["recall"])
        f1s.append(m["f1"])

    def pct_ci(arr):
        arr = np.array(arr)
        return float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))

    return {"precision_ci": pct_ci(precs), "recall_ci": pct_ci(recs), "f1_ci": pct_ci(f1s)}


def kappa(a, b):
    a = pd.Series(a).fillna(0).astype(int)
    b = pd.Series(b).fillna(0).astype(int)
    po = (a == b).mean()
    pa = a.mean()
    pb = b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else float("nan")


def score_annotations(out_dir: Path):
    cats = ["scheduling", "transportation", "documentation", "authorization"]
    a_path = out_dir / "annotation_sample_annotatorA.csv"
    b_path = out_dir / "annotation_sample_annotatorB.csv"
    if not a_path.exists() or not b_path.exists():
        raise FileNotFoundError("Annotator files not found in output dir.")
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)

    # Recompute rule flags with the current patterns to reflect rule iterations.
    for df in (a, b):
        df_flags = flag_notes(df[["text"]].rename(columns={"text": "text"}))
        for cat in cats:
            df[f"{cat}_flag"] = df_flags[f"{cat}_flag"]
    if set(a["Encounter ID"]) != set(b["Encounter ID"]):
        raise ValueError("Annotator files have different encounter sets.")
    merged = a.merge(b, on="Encounter ID", suffixes=("_A", "_B"), how="inner", validate="one_to_one")
    summary = {"per_annotator": {}, "kappa": {}, "adjudicated_vs_rules": {}, "disagreements": {}}
    for label, df in [("annotatorA", a), ("annotatorB", b)]:
        metrics = {}
        for cat in cats:
            pred = df[f"{cat}_flag"]
            gold = df[f"{cat}_gold"]
            metrics[cat] = prf(pred, gold)
        summary["per_annotator"][label] = metrics
    for cat in cats:
        summary["kappa"][cat] = kappa(merged[f"{cat}_gold_A"], merged[f"{cat}_gold_B"])

    # Adjudication: hybrid rule (presence wins if rule flag present; blank -> needs review)
    rows = []
    review_counts = defaultdict(int)
    for _, row in merged.iterrows():
        out = {
            "Encounter ID": row["Encounter ID"],
            "WaymarkId": row["WaymarkId_A"],
            "patient_id": row["patient_id_A"],
            "dateOfEncounter": row["dateOfEncounter_A"],
            "text": row["text_A"],
            "scheduling_flag": row["scheduling_flag_A"],
            "transportation_flag": row["transportation_flag_A"],
            "documentation_flag": row["documentation_flag_A"],
            "authorization_flag": row["authorization_flag_A"],
            "stratum": row["stratum_A"],
            "comment_A": row.get("comment_A", ""),
            "comment_B": row.get("comment_B", ""),
            "annotator_id_A": row.get("annotator_id_A", ""),
            "annotator_id_B": row.get("annotator_id_B", ""),
        }
        needs_review = False
        for cat in cats:
            a_val = row[f"{cat}_gold_A"]
            b_val = row[f"{cat}_gold_B"]
            if pd.isna(a_val) and pd.isna(b_val):
                out[f"{cat}_gold"] = np.nan
                needs_review = True
                review_counts[cat] += 1
            elif a_val == b_val:
                out[f"{cat}_gold"] = a_val
            else:
                flag = row[f"{cat}_flag_A"] or row[f"{cat}_flag_B"]
                out[f"{cat}_gold"] = 1.0 if flag else 0.0
        out["needs_review"] = needs_review
        rows.append(out)
    adj = pd.DataFrame(rows)
    adj.to_csv(out_dir / "annotation_sample_adjudicated.csv", index=False)
    summary["disagreements"] = {
        "per_category_needing_review": review_counts,
        "rows_needing_review": int(adj["needs_review"].sum()),
    }
    # Metrics vs rules on resolved rows only
    metrics_adj = {}
    for cat in cats:
        mask = adj[f"{cat}_gold"].notna()
        metrics_adj[cat] = prf(adj.loc[mask, f"{cat}_flag"], adj.loc[mask, f"{cat}_gold"])
    summary["adjudicated_vs_rules"] = metrics_adj
    with open(out_dir / "annotation_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "real_inputs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "notebooks" / "admin_friction" / "outputs")
    parser.add_argument("--make-annotation-sample", action="store_true", help="Regenerate annotation_sample.csv")
    parser.add_argument("--score-annotations", action="store_true", help="Compute metrics and adjudicated file from annotator CSVs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    notes, elig, mpm, attr, hosp, outcomes = load_core_data(args.data_dir)
    notes = build_linked_notes(notes, elig, mpm, attr)

    # Train ML classifiers on adjudicated corpus
    annot_adjudicated = args.output_dir / "annotation_sample_adjudicated.csv"
    annot_a = args.output_dir / "annotation_sample_annotatorA.csv"
    annot_b = args.output_dir / "annotation_sample_annotatorB.csv"
    if not annot_adjudicated.exists() or not annot_a.exists() or not annot_b.exists():
        raise FileNotFoundError("Annotation files not found. Ensure adjudicated and raw annotator files exist.")

    # Pre-adjudication kappa
    annotA = pd.read_csv(annot_a)
    annotB = pd.read_csv(annot_b)
    cats = ["scheduling", "transportation", "documentation", "authorization"]
    pre_kappa = {
        cat: kappa(annotA[f"{cat}_gold"].fillna(0), annotB[f"{cat}_gold"].fillna(0))
        for cat in cats
    }

    vectorizer, models, holdout_metrics, roc_data, thresholds = train_ml_classifiers(
        annot_adjudicated, args.output_dir / "models"
    )
    notes = apply_ml(models, vectorizer, notes, thresholds=thresholds)
    # Encounter counts per patient
    encounter_counts = notes.groupby("patient_id").size().rename("encounter_count")
    notes = notes.merge(encounter_counts, on="patient_id", how="left")
    patient_total = notes["patient_id"].nunique()
    note_total = len(notes)

    # Person-time (patient-years) for incidence (3-year window)
    patient_years = patient_total * 3.0

    # Patient-level rollup for subgroup summaries
    patient_flags = notes.groupby("patient_id").agg(
        {
            "scheduling_flag": "max",
            "transportation_flag": "max",
            "documentation_flag": "max",
            "authorization_flag": "max",
            "age": "first",
            "gender": "first",
            "ethnicity": "first",
        }
    )
    # Age groups
    bins = [0, 18, 35, 50, 65, 200]
    labels = ["<18", "18-34", "35-49", "50-64", "65+"]
    patient_flags["age_group"] = pd.cut(patient_flags["age"], bins=bins, labels=labels, right=False)

    # ROC curves for supplement
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        for ax, cat in zip(axes, ["scheduling", "transportation", "documentation", "authorization"]):
            roc = roc_data.get(cat)
            if roc:
                ax.plot(roc["fpr"], roc["tpr"], label=f"AUC={roc['auc']:.2f}")
                ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
                ax.set_title(cat.title())
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "roc_curves.png", dpi=300)
        plt.close(fig)
    except Exception:
        plt.close("all")
    notes = compute_time_cost(notes)
    notes = flag_conditions(notes)

    # Counts
    category_counts = {cat: int(notes[f"{cat}_flag"].sum()) for cat in cats}
    category_patients = {cat: int(notes.loc[notes[f"{cat}_flag"], "patient_id"].nunique()) for cat in cats}
    category_incidence = {
        cat: (count / patient_years * 100) if patient_years > 0 else float("nan") for cat, count in category_counts.items()
    }
    # Condition prevalence (note-derived)
    cond_cols = [c for c in notes.columns if c.startswith("cond_")]
    condition_patients = {c: int(notes.loc[notes[c], "patient_id"].nunique()) for c in cond_cols}

    def subgroup_prevalence(df, group_col):
        out = {}
        for grp, grp_df in df.groupby(group_col):
            denom = len(grp_df)
            if denom == 0:
                continue
            out[str(grp)] = {
                cat: float(grp_df[f"{cat}_flag"].sum()) / denom for cat in ["scheduling", "transportation", "documentation", "authorization"]
            }
        return out

    subgroup_gender = subgroup_prevalence(patient_flags, "gender")
    subgroup_ethnicity = subgroup_prevalence(patient_flags, "ethnicity")
    subgroup_age = subgroup_prevalence(patient_flags, "age_group")
    # Time cost
    total_minutes = float(notes["friction_minutes"].sum())
    total_time_cost = float(notes["time_cost"].sum())
    # Upper-bound wage sensitivity (median wage approximation $28/hr)
    total_time_cost_upper = float(notes["time_cost_upper"].sum())
    time_cost_per_patient = total_time_cost / patient_total if patient_total > 0 else float("nan")
    time_cost_per_patient_year = total_time_cost / patient_years if patient_years > 0 else float("nan")
    time_cost_per_patient_upper = total_time_cost_upper / patient_total if patient_total > 0 else float("nan")
    time_cost_per_patient_year_upper = total_time_cost_upper / patient_years if patient_years > 0 else float("nan")

    # Bootstrap CIs for time cost (patient-level resampling)
    patient_costs = notes.groupby("patient_id")[["time_cost", "time_cost_upper"]].sum()
    boot_totals = []
    boot_totals_upper = []
    rng = np.random.default_rng(13)
    n_boot = 1000
    ids = patient_costs.index.to_numpy()
    for _ in range(n_boot):
        sample_ids = rng.choice(ids, size=len(ids), replace=True)
        pc = patient_costs.loc[sample_ids]
        boot_totals.append(pc["time_cost"].sum())
        boot_totals_upper.append(pc["time_cost_upper"].sum())
    def pct_ci(arr):
        arr = np.array(arr)
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    time_cost_ci = pct_ci(boot_totals)
    time_cost_upper_ci = pct_ci(boot_totals_upper)

    # Ethnicity-stratified time costs (per patient)
    eth_map = notes.groupby("patient_id")["ethnicity"].first()
    patient_costs = patient_costs.join(eth_map)
    time_cost_by_eth = {}
    for eth, df_eth in patient_costs.groupby("ethnicity"):
        if len(df_eth) == 0:
            continue
        total_eth = float(df_eth["time_cost"].sum())
        total_eth_upper = float(df_eth["time_cost_upper"].sum())
        n_eth = len(df_eth)
        time_cost_by_eth[eth] = {
            "patients": n_eth,
            "time_cost_total": total_eth,
            "time_cost_per_patient": total_eth / n_eth,
            "time_cost_per_patient_per_year": (total_eth / n_eth) / 3,
            "time_cost_total_upper": total_eth_upper,
            "time_cost_per_patient_upper": total_eth_upper / n_eth,
            "time_cost_per_patient_per_year_upper": (total_eth_upper / n_eth) / 3,
        }

    # Acute escalation using risk difference (14-day window, scheduling exposure)
    acute_increment = estimate_acute_increment(outcomes)
    per_event_escalation = max(acute_increment - 150, 0)
    hosp = hosp.copy()
    hosp["admit_dt"] = pd.to_datetime(hosp["admit_date"], errors="coerce")
    if str(hosp["admit_dt"].dtype).startswith("datetime64[ns,"):
        hosp["admit_dt"] = hosp["admit_dt"].dt.tz_localize(None)
    hosp = hosp.dropna(subset=["patient_id", "admit_dt"])

    # Index dates and exposure
    first_note = notes.groupby("patient_id")["note_date"].min()
    sched_first = notes.loc[notes["scheduling_flag"], ["patient_id", "note_date"]].groupby("patient_id")["note_date"].min()
    exposure = first_note.to_frame("first_note").join(sched_first.rename("sched_date"), how="left")
    exposure["exposed"] = ~exposure["sched_date"].isna()
    exposure["index_date"] = exposure["sched_date"].fillna(exposure["first_note"])

    # Outcome: ED within 14 days of index
    hosp_ed = hosp.copy()
    hosp_ed = hosp_ed.merge(exposure[["index_date"]], left_on="patient_id", right_index=True, how="left")
    hosp_ed = hosp_ed.dropna(subset=["index_date"])
    hosp_ed["within_14"] = (hosp_ed["admit_dt"] >= hosp_ed["index_date"]) & (
        hosp_ed["admit_dt"] <= hosp_ed["index_date"] + pd.Timedelta(days=14)
    )
    ed_within = hosp_ed[hosp_ed["within_14"]].groupby("patient_id").size()
    exposure["ed14"] = exposure.index.map(ed_within).fillna(0)
    exposure["ed14_flag"] = exposure["ed14"] > 0

    # Covariates for propensity
    covars = notes[["patient_id", "age", "gender", "ethnicity"]].drop_duplicates("patient_id").set_index("patient_id")
    covars["gender"] = covars["gender"].fillna("Unknown")
    covars["ethnicity"] = covars["ethnicity"].fillna("Unknown")
    covars["age"] = covars["age"].fillna(covars["age"].median())
    # Prior ED before index
    prior = (
        hosp.merge(exposure[["index_date"]].reset_index(), on="patient_id", how="inner")
        .loc[lambda d: d["admit_dt"] < d["index_date"]]
        .groupby("patient_id")
        .size()
    )
    covars["prior_ed"] = covars.index.map(prior).fillna(0)
    # Assemble dataset
    df_ps = exposure.join(covars, how="left")
    df_ps = df_ps.dropna(subset=["age"])
    # Encode gender/ethnicity
    X_ps = pd.concat(
        [
            df_ps[["age", "prior_ed"]],
            pd.get_dummies(df_ps["gender"], prefix="gender"),
            pd.get_dummies(df_ps["ethnicity"], prefix="eth"),
        ],
        axis=1,
    )
    y_ps = df_ps["exposed"].astype(int)
    # Fit propensity model
    ps_model = LogisticRegression(max_iter=200, n_jobs=4)
    ps_model.fit(X_ps, y_ps)
    ps = ps_model.predict_proba(X_ps)[:, 1]
    df_ps["ps"] = ps
    df_ps["weight"] = np.where(df_ps["exposed"], 1 / df_ps["ps"].clip(1e-3, 0.999), 1 / (1 - df_ps["ps"]).clip(1e-3, 0.999))

    # Weighted risk difference (14-day) for reporting
    exposed_mask = df_ps["exposed"]
    rd = (
        (df_ps.loc[exposed_mask, "weight"] * df_ps.loc[exposed_mask, "ed14_flag"]).sum()
        / df_ps.loc[exposed_mask, "weight"].sum()
        - (df_ps.loc[~exposed_mask, "weight"] * df_ps.loc[~exposed_mask, "ed14_flag"]).sum()
        / df_ps.loc[~exposed_mask, "weight"].sum()
    )
    attributable_events_rd = rd * exposed_mask.sum()

    # Survival analysis (landmark)
    km_path = args.output_dir / "kaplan_meier_friction.png"
    hr_primary, ci_primary, surv_df, cph_summary, lr_p, ph_p = survival_analysis(notes, hosp, km_path, outcome="combined")
    # Secondary endpoints
    hr_ed, ci_ed, _, _, lr_p_ed, ph_p_ed = survival_analysis(notes, hosp, km_path.with_name("kaplan_meier_friction_ed.png"), outcome="ed")
    hr_inp, ci_inp, _, _, lr_p_inp, ph_p_inp = survival_analysis(notes, hosp, km_path.with_name("kaplan_meier_friction_inpatient.png"), outcome="inpatient")
    # Sensitivity to landmark windows (primary outcome)
    sensitivity_landmarks = {}
    for lm in [90, 180, 365]:
        hr_lm, ci_lm, _, _, lr_p_lm, ph_p_lm = survival_analysis(notes, hosp, km_path.with_name(f"kaplan_meier_friction_{lm}.png"), landmark_days=lm, outcome="combined")
        sensitivity_landmarks[lm] = {"hr": hr_lm, "ci": ci_lm, "logrank_p": lr_p_lm, "ph_p": ph_p_lm}

    # Attributable fraction and escalation cost
    af = (hr_primary - 1) / hr_primary if hr_primary and hr_primary > 0 else 0
    # Count ED events post-landmark in exposed group
    ed_post = surv_df.loc[surv_df["sched_barrier"], "event"].sum()
    aec_cost = af * ed_post * per_event_escalation

    # Minimum friction waste
    w_min = total_time_cost + aec_cost

    # Write outputs
    summary = {
        "note_total": note_total,
        "patient_total": patient_total,
        "category_counts": category_counts,
        "category_patients": category_patients,
        "category_incidence_per_100py": category_incidence,
        "condition_patients": condition_patients,
        "subgroup_prevalence_gender": subgroup_gender,
        "subgroup_prevalence_ethnicity": subgroup_ethnicity,
        "subgroup_prevalence_age": subgroup_age,
        "time_cost_total_usd": total_time_cost,
        "time_cost_total_usd_upper": total_time_cost_upper,
        "time_cost_per_patient": time_cost_per_patient,
        "time_cost_per_patient_per_year": time_cost_per_patient_year,
        "time_cost_per_patient_upper": time_cost_per_patient_upper,
        "time_cost_per_patient_per_year_upper": time_cost_per_patient_year_upper,
        "time_cost_ci": time_cost_ci,
        "time_cost_upper_ci": time_cost_upper_ci,
        "time_cost_by_ethnicity": time_cost_by_eth,
        "per_event_acute_increment": float(acute_increment),
        "per_event_escalation_cost": float(per_event_escalation),
        "risk_difference_ed14": float(rd),
        "attributable_events_ed14": float(attributable_events_rd),
        "aec_cost": float(aec_cost),
        "w_min_total_usd": float(w_min),
        "sched_barrier_patients": int(category_patients["scheduling"]),
        "patient_years": float(patient_years),
        "cox_hr_sched_primary": hr_primary,
        "cox_ci_sched_primary": ci_primary,
        "cox_logrank_p_primary": lr_p,
        "cox_ph_p_sched_primary": ph_p,
        "cox_hr_sched_ed": hr_ed,
        "cox_ci_sched_ed": ci_ed,
        "cox_logrank_p_ed": lr_p_ed,
        "cox_ph_p_sched_ed": ph_p_ed,
        "cox_hr_sched_inpatient": hr_inp,
        "cox_ci_sched_inpatient": ci_inp,
        "cox_logrank_p_inpatient": lr_p_inp,
        "cox_ph_p_sched_inpatient": ph_p_inp,
        "survival_sensitivity": sensitivity_landmarks,
        "ml_holdout_metrics": holdout_metrics,
        "pre_adjudication_kappa": pre_kappa,
        "ml_threshold": threshold,
        "attributable_fraction_sched": float(af),
    }
    with open(args.output_dir / "friction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save events file with ML flags and probabilities
    barriers = notes[
        [
            "patient_id",
            "note_date",
            "Encounter ID",
            "WaymarkId",
            "scheduling_flag",
            "transportation_flag",
            "documentation_flag",
            "authorization_flag",
            "scheduling_prob",
            "transportation_prob",
            "documentation_prob",
            "authorization_prob",
        ]
    ].rename(columns={"note_date": "note_dt"})
    barriers.to_parquet(args.output_dir / "friction_events.parquet", index=False)

    # Optional artifacts
    if args.make_annotation_sample:
        n = make_annotation_sample(flag_notes(notes.copy()), args.output_dir / "annotation_sample.csv")
        print(f"Annotation sample written: {n} rows")
    if args.score_annotations:
        metrics = score_annotations(args.output_dir)
        print("Annotation metrics:", json.dumps(metrics, indent=2))

    # Save Cox model coefficients if available
    if cph_summary is not None:
        cph_summary.to_csv(args.output_dir / "cox_model_summary.csv", index=False)

    # Narrative summary stub
    analysis_path = args.output_dir / "friction_analysis.md"
    if not analysis_path.exists():
        analysis_path.write_text(
            "See friction_summary.json for numeric results and kaplan_meier_friction.png for survival curves."
        )


if __name__ == "__main__":
    main()
