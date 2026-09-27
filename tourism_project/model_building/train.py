"""
Model training with experiment tracking for the Wellness Tourism pipeline.

- Loads the train/test splits produced by prep.py (downloaded as a workflow artifact)
- Builds a preprocessing + XGBoost pipeline and tunes it with GridSearchCV
- Logs every tuned parameter combination (nested runs) and the best model's
  parameters/metrics to MLflow
- Evaluates the best model on train and test data
- Saves the best model into tourism_project/deployment/ so the workflow can
  commit it to the repository for the Streamlit app
"""
import json
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
THRESHOLD = 0.5                      # probability cut-off for "will buy"
MODEL_DIR = Path("tourism_project/deployment")
MODEL_PATH = MODEL_DIR / "best_tourism_model_v1.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

# ------------------------------------------------------------------ MLflow
# The workflow starts `mlflow ui` on port 5000; locally you can point this
# at any tracking server via the MLFLOW_TRACKING_URI environment variable.
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")  # keep logs readable
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("tourism-wellness-package")

# ------------------------------------------------------------------ data
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")
print(f"Train {Xtrain.shape} | Test {Xtest.shape}")

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

# ------------------------------------------------------------------ model
# Only ~19% of customers bought a package, so up-weight the positive class
scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyper-parameter grid (72 combinations x 5 folds)
param_grid = {
    "xgbclassifier__n_estimators": [100, 200, 300],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.6, 0.8],
    "xgbclassifier__reg_lambda": [1, 5],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid_search = GridSearchCV(
    model_pipeline, param_grid, cv=cv, n_jobs=-1,
    scoring={"f1": "f1", "recall": "recall", "roc_auc": "roc_auc"},
    refit="f1",               # pick the model with the best precision/recall balance
    return_train_score=True,  # lets us compare train vs validation F1 (overfitting check)
)


def evaluate(model, X, y, prefix):
    """Return a dict of classification metrics at the chosen threshold."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    return {
        f"{prefix}_accuracy": accuracy_score(y, pred),
        f"{prefix}_precision": precision_score(y, pred),
        f"{prefix}_recall": recall_score(y, pred),
        f"{prefix}_f1": f1_score(y, pred),
        f"{prefix}_roc_auc": roc_auc_score(y, proba),
    }, pred


with mlflow.start_run(run_name="xgb-gridsearch") as parent_run:
    grid_search.fit(Xtrain, ytrain)

    # --- log every tuned parameter set as a nested run
    res = grid_search.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"candidate-{i:02d}", nested=True):
            mlflow.log_params({k.split("__")[1]: v for k, v in params.items()})
            mlflow.log_metrics({
                "cv_f1_mean": res["mean_test_f1"][i],
                "cv_f1_std": res["std_test_f1"][i],
                "cv_train_f1_mean": res["mean_train_f1"][i],
                "cv_recall_mean": res["mean_test_recall"][i],
                "cv_roc_auc_mean": res["mean_test_roc_auc"][i],
            })

    # --- log the best parameters on the parent run
    best_params = {k.split("__")[1]: v for k, v in grid_search.best_params_.items()}
    mlflow.log_params(best_params)
    mlflow.log_params({"threshold": THRESHOLD, "cv_folds": 5,
                       "scale_pos_weight": round(float(scale_pos_weight), 3),
                       "n_candidates": len(res["params"])})
    best_idx = grid_search.best_index_
    mlflow.log_metrics({
        "best_cv_f1": grid_search.best_score_,
        "best_cv_train_f1": res["mean_train_f1"][best_idx],
        "best_cv_f1_gap": res["mean_train_f1"][best_idx] - grid_search.best_score_,
    })

    # --- evaluate the best model
    best_model = grid_search.best_estimator_
    train_metrics, _ = evaluate(best_model, Xtrain, ytrain, "train")
    test_metrics, test_pred = evaluate(best_model, Xtest, ytest, "test")
    mlflow.log_metrics({**train_metrics, **test_metrics})

    print("\nBest parameters:", best_params)
    print(f"Best CV F1: {grid_search.best_score_:.4f} "
          f"(train folds {res['mean_train_f1'][best_idx]:.4f})\n")
    print(pd.DataFrame({
        "train": {k.replace("train_", ""): v for k, v in train_metrics.items()},
        "test": {k.replace("test_", ""): v for k, v in test_metrics.items()},
    }).round(4).to_string())
    print("\nTest classification report:")
    print(classification_report(ytest, test_pred, digits=4))

    # --- save the best model for deployment and log it as an artifact
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "best_params": best_params,
        "threshold": THRESHOLD,
        **{k: round(float(v), 4) for k, v in {**train_metrics, **test_metrics}.items()},
    }, indent=2))
    mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
    mlflow.log_artifact(str(METRICS_PATH), artifact_path="model")
    print(f"Saved best model -> {MODEL_PATH}")
    print(f"MLflow run id: {parent_run.info.run_id}")
