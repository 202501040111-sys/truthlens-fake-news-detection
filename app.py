from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from pathlib import Path
import pickle, json, re
import pandas as pd

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
UPLOAD_DIR = BASE / "uploads"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

# FIX: make sure the folders exist before any upload or save happens.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

app.secret_key = "replace-this-secret-for-deployment"


def model_ready():
    return MODEL_PATH.exists() and VECTORIZER_PATH.exists()

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None

def clean_text(text):
    text = str(text or "")
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    news_text = ""
    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()
        if len(news_text) < 20:
            flash("Please enter at least 20 characters of news text.", "error")
        elif not model_ready():
            flash("No trained model found. Upload your dataset and train the model first in the Training page.", "error")
        else:
            try:
                model, vectorizer = load_model()
                vector = vectorizer.transform([news_text])
                prediction = str(model.predict(vector)[0]).title()
                confidence = None
                if hasattr(model, "predict_proba"):
                    confidence = float(max(model.predict_proba(vector)[0])) * 100
                result = {"label": prediction, "confidence": confidence}
            except Exception as exc:
                flash(f"Prediction failed: {exc}", "error")
    return render_template("index.html", result=result, news_text=news_text,
                           ready=model_ready(), metrics=load_metrics())

@app.route("/training", methods=["GET", "POST"])
def training():
    if request.method == "POST":
        mode = request.form.get("mode")
        try:
            # Safety net in case the folder was deleted while the app was running.
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

            if mode == "isot":
                fake_file = request.files.get("fake_file")
                real_file = request.files.get("real_file")
                if not fake_file or not real_file or not fake_file.filename or not real_file.filename:
                    flash("Please select both Fake.csv and True.csv files.", "error")
                    return redirect(url_for("training"))
                fake_path = UPLOAD_DIR / "Fake.csv"
                real_path = UPLOAD_DIR / "True.csv"
                fake_file.save(fake_path)
                real_file.save(real_path)
                train_from_isot(fake_path, real_path)
            else:
                dataset_file = request.files.get("dataset")
                if not dataset_file or not dataset_file.filename:
                    flash("Choose a CSV dataset file.", "error")
                    return redirect(url_for("training"))
                dataset_path = UPLOAD_DIR / "dataset.csv"
                dataset_file.save(dataset_path)
                train_from_csv(dataset_path)
            flash("Training completed! Models were evaluated and the best model was saved.", "success")
        except Exception as exc:
            flash(f"Training error: {exc}", "error")
        return redirect(url_for("training"))
    return render_template("training.html", metrics=load_metrics(), ready=model_ready())

def train_pipeline(df):
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    # Standardise columns and remove unusable rows.
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    text_col = next((c for c in ["text", "news", "article", "content", "title"] if c in df.columns), None)
    label_col = next((c for c in ["label", "class", "target", "category"] if c in df.columns), None)
    if not text_col or not label_col:
        raise ValueError("CSV must have a text/content/title column and a label/class column.")
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[text_col].str.len() >= 20]
    df = df.drop_duplicates(subset=[text_col])
    labels = df[label_col].str.lower().replace({
        "0": "fake", "1": "real", "false": "fake", "true": "real",
        "fake news": "fake", "real news": "real", "misleading": "fake"
    })
    # Keep the two-category project scope.
    valid = labels.isin(["fake", "real"])
    df = df[valid]
    labels = labels[valid]
    if labels.nunique() != 2 or len(df) < 20:
        raise ValueError("Need at least 20 labelled rows containing both Fake and Real classes.")
    if labels.value_counts().min() < 2:
        raise ValueError("Each class must contain at least two examples.")

    X = df[text_col].values
    y = labels.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced")
    }
    all_metrics = {}
    best_name, best_score, best_model, best_vectorizer = None, -1, None, None
    for name, clf in candidates.items():
        vec = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1, 2), min_df=1)
        train_vec = vec.fit_transform(X_train)
        test_vec = vec.transform(X_test)
        clf.fit(train_vec, y_train)
        pred = clf.predict(test_vec)
        scores = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, pos_label="fake", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, pos_label="fake", zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, pos_label="fake", zero_division=0)), 4),
        }
        all_metrics[name] = scores
        if scores["f1"] > best_score:
            best_name, best_score, best_model, best_vectorizer = name, scores["f1"], clf, vec

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(best_vectorizer, f)
    summary = {
        "best_model": best_name,
        "metrics": all_metrics,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_counts": {str(k): int(v) for k, v in labels.value_counts().items()},
        "positive_class_for_precision_recall_f1": "fake",
        "note": "Metrics are from a random stratified holdout. They are not proof of factual truth or performance on future news."
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def train_from_csv(path):
    train_pipeline(pd.read_csv(path))

def train_from_isot(fake_path, real_path):
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)
    fake["label"] = "fake"
    real["label"] = "real"
    combined = pd.concat([fake, real], ignore_index=True)
    # Prefer article text; use title if text is missing.
    if "text" in combined.columns and "title" in combined.columns:
        combined["text"] = combined["title"].fillna("") + " " + combined["text"].fillna("")
    elif "text" not in combined.columns and "title" in combined.columns:
        combined["text"] = combined["title"]
    train_pipeline(combined)

@app.route("/api/status")
def status():
    return jsonify({"trained": model_ready(), "metrics": load_metrics()})

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)