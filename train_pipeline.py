"""
=============================================================================
AI Student Impact — ML Pipeline
=============================================================================
Targets
  • Classification  : Burnout_Risk_Level  (Low / Medium / High)
  • Regression      : Post_Semester_GPA

Models
  • XGBoost Classifier  (burnout_model.pkl)
  • XGBoost Regressor   (gpa_model.pkl)

Artefacts saved (ready for a Streamlit app)
  burnout_model.pkl   gpa_model.pkl
  label_encoder.pkl   feature_pipeline.pkl
=============================================================================
"""

# ── Standard / third-party imports ──────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import joblib

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import (LabelEncoder, OrdinalEncoder,
                                      StandardScaler)
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.metrics          import (classification_report,
                                      r2_score, mean_absolute_error)
from xgboost import XGBClassifier, XGBRegressor

# ── Paths ────────────────────────────────────────────────────────────────────
# Get script directory for robust path resolution
SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_PATH           = os.path.join(SCRIPT_DIR, "data/ai_student_impact_dataset.csv")
BURNOUT_MODEL_PATH  = os.path.join(SCRIPT_DIR, "models/burnout_model.pkl")
GPA_MODEL_PATH      = os.path.join(SCRIPT_DIR, "models/gpa_model.pkl")
LABEL_ENC_PATH      = os.path.join(SCRIPT_DIR, "models/label_encoder.pkl")
FEAT_PIPE_PATH      = os.path.join(SCRIPT_DIR, "models/feature_pipeline.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load the CSV and drop columns that carry no predictive signal."""
    df = pd.read_csv(path)
    print(f"[load]  Shape = {df.shape}")
    print(f"[load]  Null values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Student_ID is a surrogate key — not a feature
    df = df.drop(columns=["Student_ID"])

    # Cast boolean column to int so XGBoost is happy
    df["Paid_Subscription"] = df["Paid_Subscription"].astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Ordinal categoricals — meaningful rank order exists
ORDINAL_FEATURES = {
    "Year_of_Study": ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
    "Prompt_Engineering_Skill": ["Beginner", "Intermediate", "Advanced"],
}

# Nominal categoricals — no rank order; use One-Hot Encoding
NOMINAL_FEATURES = [
    "Major_Category",
    "Primary_Use_Case",
    "Institutional_Policy",
]

# Numeric features (including bool-cast column)
NUMERIC_FEATURES = [
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Tool_Diversity",
    "Paid_Subscription",
    "Traditional_Study_Hours",
    "Perceived_AI_Dependency",
    "Anxiety_Level_During_Exams",
    "Skill_Retention_Score",
]

# Targets
TARGET_CLASS = "Burnout_Risk_Level"   # Classification
TARGET_REG   = "Post_Semester_GPA"    # Regression


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_pipeline() -> ColumnTransformer:
    """
    Returns a ColumnTransformer that:
      • OrdinalEncodes ordered categoricals (preserves rank signal)
      • OneHotEncodes nominal categoricals
      • StandardScales numeric features
    """
    ordinal_pipe = Pipeline([
        ("enc", OrdinalEncoder(
            categories=[ORDINAL_FEATURES[c] for c in ORDINAL_FEATURES],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    from sklearn.preprocessing import OneHotEncoder
    nominal_pipe = Pipeline([
        ("enc", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ])

    numeric_pipe = Pipeline([
        ("scaler", StandardScaler()),
    ])

    ct = ColumnTransformer(transformers=[
        ("ordinal", ordinal_pipe,  list(ORDINAL_FEATURES.keys())),
        ("nominal", nominal_pipe,  NOMINAL_FEATURES),
        ("numeric", numeric_pipe,  NUMERIC_FEATURES),
    ], remainder="drop")

    return ct


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCODE CLASSIFICATION TARGET
# ─────────────────────────────────────────────────────────────────────────────
def encode_target(series: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """
    LabelEncode 'Burnout_Risk_Level'.
    XGBoost requires integer class indices (0-based).
    """
    le = LabelEncoder()
    y  = le.fit_transform(series)
    print(f"[target] Burnout classes: {list(le.classes_)}")
    return y, le


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN / EVALUATE — CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def train_classifier(X_train, X_val, y_train, y_val,
                     le: LabelEncoder) -> XGBClassifier:
    """
    Train XGBClassifier with multi-class objective.
    Prints classification report + top-15 feature importances.
    """
    clf = XGBClassifier(
        objective        = "multi:softprob",
        num_class        = len(le.classes_),
        n_estimators     = 500,
        learning_rate    = 0.05,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        use_label_encoder= False,
        eval_metric      = "mlogloss",
        random_state     = 42,
        n_jobs           = -1,
    )

    clf.fit(
        X_train, y_train,
        eval_set              = [(X_val, y_val)],
        verbose               = 100,
    )

    y_pred = clf.predict(X_val)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT — Burnout_Risk_Level")
    print("="*60)
    print(classification_report(y_val, y_pred,
                                target_names=le.classes_))

    # Feature importance
    _print_feature_importances(clf)

    return clf


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN / EVALUATE — REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
def train_regressor(X_train, X_val, y_train, y_val) -> XGBRegressor:
    """
    Train XGBRegressor to predict Post_Semester_GPA.
    Prints R² and MAE.
    """
    reg = XGBRegressor(
        objective        = "reg:squarederror",
        n_estimators     = 500,
        learning_rate    = 0.05,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        eval_metric      = "rmse",
        random_state     = 42,
        n_jobs           = -1,
    )

    reg.fit(
        X_train, y_train,
        eval_set = [(X_val, y_val)],
        verbose  = 100,
    )

    y_pred = reg.predict(X_val)
    r2  = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    print("\n" + "="*60)
    print("REGRESSION METRICS — Post_Semester_GPA")
    print("="*60)
    print(f"  R²  Score : {r2:.4f}")
    print(f"  MAE       : {mae:.4f}")
    print("="*60)

    return reg


# ─────────────────────────────────────────────────────────────────────────────
# 7. HELPER — FEATURE IMPORTANCE PRINTER
# ─────────────────────────────────────────────────────────────────────────────
def _print_feature_importances(model, top_n: int = 15) -> None:
    """Derive readable feature names and print top-N importances."""
    # Reconstruct feature names from the ColumnTransformer
    # (called after fit on the pipeline, so we can inspect the transformer)
    importances = model.feature_importances_

    print(f"\nTop-{top_n} Feature Importances (gain):")
    idx = np.argsort(importances)[::-1][:top_n]
    for rank, i in enumerate(idx, 1):
        print(f"  {rank:>2}. feature[{i:>3}]  →  {importances[i]:.4f}")

    print()
    print("  (Use feature_names_out_ from the saved feature_pipeline.pkl",
          "to map indices to column names in your Streamlit app.)")


# ─────────────────────────────────────────────────────────────────────────────
# 8. SAVE ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(clf, reg, le, feat_pipe) -> None:
    """Persist all artefacts needed by the downstream Streamlit app."""
    import os
    os.makedirs("models", exist_ok=True)

    joblib.dump(clf,       BURNOUT_MODEL_PATH)
    joblib.dump(reg,       GPA_MODEL_PATH)
    joblib.dump(le,        LABEL_ENC_PATH)
    joblib.dump(feat_pipe, FEAT_PIPE_PATH)

    print("\n[save] Artefacts written:")
    print(f"  • {BURNOUT_MODEL_PATH}")
    print(f"  • {GPA_MODEL_PATH}")
    print(f"  • {LABEL_ENC_PATH}")
    print(f"  • {FEAT_PIPE_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "="*60)
    print("  AI Student Impact — ML Pipeline")
    print("="*60 + "\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_data(DATA_PATH)

    # ── Separate features from both targets ───────────────────────────────────
    feature_cols = (list(ORDINAL_FEATURES.keys()) +
                    NOMINAL_FEATURES               +
                    NUMERIC_FEATURES)

    X   = df[feature_cols]
    y_c = df[TARGET_CLASS]          # classification target (string)
    y_r = df[TARGET_REG].values     # regression target (float)

    # ── Encode classification target ──────────────────────────────────────────
    y_encoded, le = encode_target(y_c)

    # ── Train / validation split (stratified on burnout classes) ─────────────
    (X_train, X_val,
     yc_train, yc_val,
     yr_train, yr_val) = train_test_split(
        X, y_encoded, y_r,
        test_size    = 0.20,
        random_state = 42,
        stratify     = y_encoded,
    )
    print(f"\n[split] Train={len(X_train):,}  Val={len(X_val):,}")

    # ── Fit feature pipeline on training data only ────────────────────────────
    feat_pipe = build_feature_pipeline()
    X_train_t = feat_pipe.fit_transform(X_train)
    X_val_t   = feat_pipe.transform(X_val)
    print(f"[preproc] Transformed feature matrix shape: {X_train_t.shape}")

    # ── Train models ──────────────────────────────────────────────────────────
    print("\n[train] XGBoost Classifier (Burnout_Risk_Level) ...")
    clf = train_classifier(X_train_t, X_val_t, yc_train, yc_val, le)

    print("\n[train] XGBoost Regressor (Post_Semester_GPA) ...")
    reg = train_regressor(X_train_t, X_val_t, yr_train, yr_val)

    # ── Persist artefacts ─────────────────────────────────────────────────────
    save_artifacts(clf, reg, le, feat_pipe)

    print("\n✓ Pipeline complete — all artefacts ready for Streamlit.\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
