"""TF-IDF comparison models (scikit-learn pipelines) for the LLM experiments.

tfidf_lr_meta is also the transferred UC1 classifier used on UC2 and the model explained with SHAP.
BERT is trained separately in 05_bert_uc1_colab.ipynb.

Thesis: Section 3.4 (comparison models).
Used by: 03_run_uc1_baselines.py, 08_run_uc2_evaluation.py, 10_shap_beeswarm_tfidf.py.
"""
from __future__ import annotations


def _tfidf():
    """Build the shared TF-IDF vectorizer (20,000 features, 1-2 word n-grams)."""
    # Lazy import, so importing this module does not load scikit-learn.
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                           sublinear_tf=True, stop_words="english")


def build_text_model(kind: str):
    """Build a text-only pipeline: TF-IDF on "text", then a classifier.

    Args:
        kind: "lr" for Logistic Regression or "rf" for Random Forest.
    """
    from sklearn.pipeline import Pipeline
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     n_jobs=-1, random_state=42)
    else:
        raise ValueError(f"unknown kind: {kind}")
    # A ColumnTransformer lets this model take a DataFrame, like the meta model.
    from sklearn.compose import ColumnTransformer
    pre = ColumnTransformer([("text", _tfidf(), "text")], remainder="drop")
    return Pipeline([("pre", pre), ("clf", clf)])


def build_meta_model(kind: str = "lr"):
    """Build a Logistic Regression pipeline on TF-IDF text, one-hot sender_tld and scaled num_urls.

    Args:
        kind: Kept for a uniform interface; only "lr" is used.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    pre = ColumnTransformer([
        ("text", _tfidf(), "text"),
        ("tld", OneHotEncoder(handle_unknown="ignore", max_categories=40), ["sender_tld"]),
        ("num", StandardScaler(), ["num_urls"]),
    ], remainder="drop")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


# name -> (builder, needs_meta_cols); needs_meta_cols means it also uses sender_tld and num_urls.
MODELS = {
    "tfidf_lr":      (lambda: build_text_model("lr"), False),
    "tfidf_rf":      (lambda: build_text_model("rf"), False),
    "tfidf_lr_meta": (lambda: build_meta_model("lr"), True),
}
