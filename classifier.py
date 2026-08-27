"""
ML category classifier: TF-IDF + Logistic Regression, trained on the
user's own historical (note -> category) expense data stored in SQLite.

This is the "gets smarter as it learns" half of the hybrid system:

Early on, the LLM's category guess is used as-is.

Once enough labeled history accumulates, this classifier is trained and
its predictions (when confident) can override the LLM/regex guess because
it reflects the user's own actual categorization habits.
"""

import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity

from database import get_all_expenses


MODEL_PATH = "category_model.joblib"

MIN_SAMPLES_PER_CLASS = 3

CONFIDENCE_THRESHOLD = 0.55

SIMILARITY_THRESHOLD = 0.12


def _load_training_data():
    """
    Load expense training data from SQLite.

    The classifier learns from:
        note -> category

    Returns:
        pandas DataFrame or None
    """

    df = get_all_expenses()

    if df is None or df.empty:
        return None

    # Make sure required columns exist.
    if "note" not in df.columns or "category" not in df.columns:
        return None

    # Remove rows where note/category is missing.
    df = df.dropna(
        subset=["note", "category"]
    )

    if df.empty:
        return None

    # Force both columns to strings.
    df["note"] = df["note"].astype(str)
    df["category"] = df["category"].astype(str)

    # Remove empty notes.
    df = df[
        df["note"].str.strip() != ""
    ]

    if df.empty:
        return None

    return df


def can_train():
    """
    Check whether there is enough labeled data to train
    a useful category classifier.

    A category needs at least MIN_SAMPLES_PER_CLASS
    examples.

    At least two categories must be eligible.
    """

    df = _load_training_data()

    if df is None:
        return False, "No expense data yet."

    counts = df["category"].value_counts()

    eligible = counts[
        counts >= MIN_SAMPLES_PER_CLASS
    ]

    if len(eligible) < 2:

        return False, (
            f"Need at least 2 categories with "
            f"{MIN_SAMPLES_PER_CLASS}+ examples each. "
            f"Currently: {dict(counts)}"
        )

    return True, (
        f"{len(eligible)} categories ready to train "
        f"({int(counts.sum())} total rows)."
    )


def train_classifier(
    model_path=MODEL_PATH
):
    """
    Train or retrain the category classifier
    using expense history stored in SQLite.

    Returns:
        Dictionary containing training statistics.
    """

    df = _load_training_data()

    if df is None:
        raise ValueError(
            "No training data available."
        )

    counts = df["category"].value_counts()

    valid_categories = counts[
        counts >= MIN_SAMPLES_PER_CLASS
    ].index

    df = df[
        df["category"].isin(valid_categories)
    ]

    if df["category"].nunique() < 2:

        raise ValueError(
            "Need at least 2 categories with "
            "enough examples to train."
        )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1
            )
        ),

        (
            "clf",
            LogisticRegression(
                max_iter=1000
            )
        ),
    ])

    pipeline.fit(
        df["note"],
        df["category"]
    )

    # Save the trained model and training notes.
    joblib.dump(
        {
            "pipeline": pipeline,
            "train_notes": df["note"].tolist()
        },
        model_path
    )

    # Quick cross-validation accuracy estimate.
    cv_accuracy = None

    try:

        min_class_count = (
            df["category"]
            .value_counts()
            .min()
        )

        cv_folds = max(
            2,
            min(3, min_class_count)
        )

        scores = cross_val_score(
            pipeline,
            df["note"],
            df["category"],
            cv=cv_folds
        )

        cv_accuracy = float(
            scores.mean()
        )

    except Exception:
        # Training can still succeed even if
        # cross-validation isn't possible.
        pass

    return {
        "n_samples": len(df),

        "n_categories": int(
            df["category"].nunique()
        ),

        "categories": sorted(
            df["category"]
            .unique()
            .tolist()
        ),

        "cv_accuracy": cv_accuracy,
    }


def load_model(
    model_path=MODEL_PATH
):
    """
    Load the trained classifier if it exists.
    """

    if os.path.exists(model_path):

        return joblib.load(
            model_path
        )

    return None


def predict_category(
    note: str,
    model_path=MODEL_PATH
):
    """
    Predict a category from an expense note.

    Returns:
        (predicted_category, confidence)

    Returns:
        (None, 0.0)

    when:

    - no model exists
    - note is empty
    - note is unfamiliar
    """

    bundle = load_model(
        model_path
    )

    if bundle is None or note is None:

        return None, 0.0

    note = str(note)

    if not note.strip():

        return None, 0.0

    pipeline = bundle["pipeline"]

    train_notes = bundle["train_notes"]

    # Get probabilities for each known category.
    proba = pipeline.predict_proba(
        [note]
    )[0]

    classes = pipeline.classes_

    best_idx = proba.argmax()

    confidence = float(
        proba[best_idx]
    )

    predicted = classes[best_idx]

    # Similarity guard.
    tfidf = pipeline.named_steps[
        "tfidf"
    ]

    query_vec = tfidf.transform(
        [note]
    )

    train_vecs = tfidf.transform(
        train_notes
    )

    max_similarity = (
        cosine_similarity(
            query_vec,
            train_vecs
        ).max()
        if train_vecs.shape[0]
        else 0.0
    )

    # If the note is too different from
    # anything the model has seen, don't guess.
    if (
        max_similarity
        < SIMILARITY_THRESHOLD
    ):

        return None, 0.0

    return predicted, confidence
