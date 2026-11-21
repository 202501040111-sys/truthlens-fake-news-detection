# TruthLens – Fake News Detection Website

A Flask web application for the college project **AI-Based Fake News Detection Using Natural Language Processing**.

## Features
- Responsive web UI
- Paste headline/article and predict Fake or Real
- Dataset training page
- Upload ISOT dataset (`Fake.csv` and `True.csv`) or one labelled CSV
- TF-IDF unigram + bigram text features
- Trains Naive Bayes, Logistic Regression and Linear SVM
- Compares accuracy, precision, recall and F1-score
- Saves the best F1 model and TF-IDF vectorizer locally
- Shows dataset counts and metrics from actual training
- Empty/short text validation and limitations notice

## 1. Install
Install Python 3.10+ then open a terminal in this folder:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Mac/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 2. Train model
Start the website:
```bash
python app.py
```
Open http://127.0.0.1:5000 and choose **Train Model**.

### ISOT dataset
Upload both files:
- `Fake.csv`
- `True.csv`

The app labels records according to which file they came from and combines `title` and `text` when available.

### Single CSV
CSV must contain:
- one text column such as `text`, `news`, `article`, `content` or `title`
- one label column such as `label`, `class`, `target` or `category`

Labels accepted: `Fake`, `Real`, `0` (Fake), `1` (Real), `false` (Fake), `true` (Real).

Training requires at least 20 usable labelled rows and both classes.

## 3. Use the detector
Go back to Detector, paste at least 20 characters, and click Analyze text.

## Project pipeline
Dataset -> Cleaning -> Train/Test Split -> TF-IDF -> Naive Bayes / Logistic Regression / Linear SVM -> Compare metrics -> Save best model -> Web prediction.

## Important for your report
Only report metric values that the app actually calculates after you train it on your dataset. This is a text-pattern screening prototype; it does **not** independently verify claims or prove truth. Performance can be affected by dataset bias, duplicate records, source/topic patterns and distribution changes.

## Troubleshooting
- If `pip` is not recognised, try `python -m pip install -r requirements.txt`.
- If port 5000 is busy, edit the final `app.run(... port=5000)` line.
- Keep the `models` folder so the saved model and vectorizer remain available.
