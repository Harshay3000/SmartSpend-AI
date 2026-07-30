"""
ML category classifier: TF-IDF + Logistic Regression, trained on the
user's own historical (note -> category) expense data.

This is the "gets smarter as it learns" half of the hybrid system:
early on, the LLM's category guess is used as-is. Once enough labeled
history accumulates, this classifier is trained and its predictions
(when confident) override the LLM/regex guess -- since it reflects the
user's own actual categorization habits, not a generic LLM guess.
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = "category_model.joblib"
MIN_SAMPLES_PER_CLASS = 3       # a category needs this many examples to be trainable
CONFIDENCE_THRESHOLD = 0.55     # below this, defer to the LLM/regex category instead
SIMILARITY_THRESHOLD = 0.12     # min TF-IDF cosine similarity to nearest training example


def _load_training_data(filepath="expenses.csv"):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return None
    df = pd.read_csv(filepath)
    if "note" not in df.columns or "category" not in df.columns:
        return None
    df = df.dropna(subset=["note", "category"])
    if df.empty:
        return None
    # Force to string dtype: if a column happens to contain only
    # numeric-looking values (e.g. a note that was just digits), pandas
    # infers it as int64/float64 on read, and .str accessor raises
    # AttributeError even though there are no NaNs left.
    df["note"] = df["note"].astype(str)
    df["category"] = df["category"].astype(str)
    df = df[df["note"].str.strip() != ""]
    return df if not df.empty else None


def can_train(filepath="expenses.csv"):
    """Check whether there's enough labeled data yet to train a useful model."""
    df = _load_training_data(filepath)
    if df is None:
        return False, "No expense data yet."

    counts = df["category"].value_counts()
    eligible = counts[counts >= MIN_SAMPLES_PER_CLASS]

    if len(eligible) < 2:
        return False, (
            f"Need at least 2 categories with {MIN_SAMPLES_PER_CLASS}+ examples "
            f"each. Currently: {dict(counts)}"
        )
    return True, f"{len(eligible)} categories ready to train ({int(counts.sum())} total rows)."


def train_classifier(filepath="expenses.csv", model_path=MODEL_PATH):
    """
    Train (or retrain) the category classifier on all available labeled
    expense history. Returns a dict of training stats for display in the UI.
    """
    df = _load_training_data(filepath)
    if df is None:
        raise ValueError("No training data available.")

    counts = df["category"].value_counts()
    valid_categories = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    df = df[df["category"].isin(valid_categories)]

    if df["category"].nunique() < 2:
        raise ValueError("Need at least 2 categories with enough examples to train.")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(df["note"], df["category"])
    # Persist the training notes alongside the pipeline so predict_category
    # can check similarity against them (see out-of-distribution guard below).
    joblib.dump({"pipeline": pipeline, "train_notes": df["note"].tolist()}, model_path)

    # quick honest accuracy estimate via cross-validation
    cv_accuracy = None
    try:
        min_class_count = df["category"].value_counts().min()
        cv_folds = max(2, min(3, min_class_count))
        scores = cross_val_score(pipeline, df["note"], df["category"], cv=cv_folds)
        cv_accuracy = float(scores.mean())
    except Exception:
        pass  # not enough data per class for CV; skip the estimate, training still succeeded

    return {
        "n_samples": len(df),
        "n_categories": int(df["category"].nunique()),
        "categories": sorted(df["category"].unique().tolist()),
        "cv_accuracy": cv_accuracy,
    }


def load_model(model_path=MODEL_PATH):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def predict_category(note: str, model_path=MODEL_PATH):
    """
    Returns (predicted_category, confidence), or (None, 0.0) if no model is
    trained, the note is empty, or the note doesn't resemble anything the
    model was actually trained on.

    Note on the similarity guard: predict_proba's confidence is a softmax
    over *known* categories only -- it always sums to 1 even for a totally
    unfamiliar note, which can look deceptively confident (especially with
    few categories). Requiring a minimum TF-IDF similarity to some real
    training example gives the model a way to effectively say "I don't
    recognize this" instead of confidently guessing among what it knows.
    """
    bundle = load_model(model_path)
    if bundle is None or note is None:
        return None, 0.0
    note = str(note)
    if not note.strip():
        return None, 0.0

    pipeline = bundle["pipeline"]
    train_notes = bundle["train_notes"]

    proba = pipeline.predict_proba([note])[0]
    classes = pipeline.classes_
    best_idx = proba.argmax()
    confidence = float(proba[best_idx])
    predicted = classes[best_idx]

    tfidf = pipeline.named_steps["tfidf"]
    query_vec = tfidf.transform([note])
    train_vecs = tfidf.transform(train_notes)
    max_similarity = cosine_similarity(query_vec, train_vecs).max() if train_vecs.shape[0] else 0.0

    if max_similarity < SIMILARITY_THRESHOLD:
        return None, 0.0

    return predicted, confidence
