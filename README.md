# ReqGuard AI

## Machine Learning and NLP-Based Software Requirement Quality and Risk Analysis

ReqGuard AI is a Computer Science / AI project applied to software requirements engineering. It analyzes natural-language software requirements before development begins and identifies requirements that may be unclear, incomplete, difficult to test, vague, semantically duplicated, or potentially conflicting.

The upgraded version combines **explainable rule-based analysis with machine learning and transformer-based NLP**. It is intentionally different from a project that evaluates AI-generated test cases: ReqGuard evaluates the **software requirements themselves**.

## Main features

- Rule-based scoring for clarity, completeness, testability, specificity, and ambiguity.
- Risk classification using four ML algorithms: Logistic Regression, Random Forest, Linear SVM, and Naive Bayes.
- Automatic model comparison using accuracy, precision, recall, F1-score, classification report, and confusion matrix.
- ML-based numerical quality-score prediction using Random Forest Regression.
- Hybrid decision system: 60% explainable rule score + 40% ML quality prediction.
- TF-IDF and cosine similarity for traditional NLP comparison.
- Optional Sentence-BERT (`all-MiniLM-L6-v2`) semantic embeddings for meaning-based similarity.
- Semantic duplicate-requirement detection.
- Potential conflict detection using semantic similarity plus negation and incompatible numeric constraints.
- K-Means clustering to group related requirements.
- Batch CSV analysis and downloadable result files.
- Streamlit dashboard and Plotly visualizations.
- SQLite storage for analysis history.
- Pytest automated tests.

## Project workflow

```text
Software Requirement(s)
        |
        v
Text / NLP Processing
   |             |
   v             v
Rule Engine    TF-IDF / Sentence-BERT
   |             |
   |        ML Classification
   |        ML Regression
   |        Semantic Similarity
   |        K-Means Clustering
   |             |
   +------ Hybrid Decision ------+
                 |
                 v
       Quality Score + Risk
                 |
                 v
     Issues + Recommendations
                 |
                 v
        Streamlit Dashboard
```

## ML components

### Risk classification
The training dataset contains requirement text and one of three risk classes: `LOW`, `MEDIUM`, or `HIGH`. ReqGuard trains and compares:

1. Logistic Regression
2. Random Forest Classifier
3. Linear SVM
4. Multinomial Naive Bayes

The classifier with the highest weighted F1-score is saved as the default risk model.

### Quality regression
A Random Forest Regressor learns to predict a quality score between 0 and 100 from requirement text. The dashboard reports MAE, RMSE, and R².

### Transformer semantics
If `sentence-transformers` is installed and the model can be loaded, ReqGuard uses `all-MiniLM-L6-v2` to create semantic sentence embeddings. On the first use, the model may need to be downloaded. If it is unavailable, the project falls back to TF-IDF so the application remains usable.

### Clustering
K-Means groups requirements using Sentence-BERT embeddings when available, otherwise TF-IDF features. This demonstrates unsupervised machine learning.

## Validation results

The current implementation was validated locally on Windows with Python 3.12 using the bundled labeled demonstration dataset.

### Automated tests

```text
8 passed
```

All eight automated tests passed successfully.

### Held-out risk-classification benchmark

The bundled 150-example dataset was split into **112 training** and **38 held-out test** requirements. Four classifiers were compared on the same split.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Random Forest | **0.9737** | **0.9757** | **0.9737** | **0.9737** |
| Linear SVM | 0.9211 | 0.9359 | 0.9211 | 0.9222 |
| Logistic Regression | 0.8684 | 0.8932 | 0.8684 | 0.8649 |
| Naive Bayes | 0.8158 | 0.8461 | 0.8158 | 0.8090 |

**Random Forest** was the best-performing classifier on this held-out split.

Its confusion matrix was:

```text
[[13, 0, 0],
 [ 0,12, 1],
 [ 0, 0,12]]
```

### Quality-regression benchmark

The Random Forest quality regressor was evaluated on the same 112/38 train/test split.

| Metric | Result |
|---|---:|
| MAE | **4.57** |
| RMSE | **5.64** |
| R² | **0.901** |

### Manual sanity checks

A measurable requirement:

```text
The API shall respond within 2 seconds.
```

was classified as **LOW risk**, with probabilities:

```text
LOW: 0.788
MEDIUM: 0.152
HIGH: 0.060
```

An ambiguous requirement:

```text
The system should work quickly.
```

was classified as **HIGH risk**, with probabilities:

```text
HIGH: 0.680
MEDIUM: 0.244
LOW: 0.076
```

The hybrid assessment for:

```text
The application should respond quickly.
```

produced a **final quality score of 57.7** and **HIGH final risk**. The rule engine detected issues including the ambiguous word `quickly`, missing context, and the absence of a measurable threshold.

These validation results demonstrate that the implemented ML, rule-based, hybrid, and testing pipelines behave consistently on the bundled demonstration benchmark. They do **not** establish production-level or broad real-world performance.

## Important academic limitation

The bundled `training_requirements.csv` is a **small synthetic demonstration dataset** generated for the prototype. Metrics produced from it show that the ML pipeline works, but they must not be presented as evidence of production-level or real-world performance. For a thesis or research study, replace or extend it with a larger manually labeled dataset and perform stronger validation.

## Installation on Windows

### Option 1: one-click helper
Double-click:

```text
run_windows.bat
```

### Option 2: command prompt / PowerShell

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The Streamlit application will usually open at `http://localhost:8501`.

## Lightweight installation without Sentence-BERT

If installing PyTorch / Sentence Transformers is difficult, install only:

```bash
pip install streamlit pandas plotly scikit-learn joblib numpy pytest
streamlit run app.py
```

The semantic functions will automatically use TF-IDF fallback mode.

## CSV format

Batch analysis expects at least a `requirement` column:

```csv
id,requirement
R001,The user shall login using email and password.
R002,The website should work quickly.
```

## Main files

- `app.py` — Streamlit dashboard
- `quality_engine.py` — explainable quality metrics
- `ambiguity.py` — ambiguity detection
- `ml_model.py` — classifiers, model comparison, and quality regression
- `semantic_analysis.py` — Sentence-BERT / TF-IDF semantic analysis
- `clustering.py` — K-Means grouping
- `hybrid.py` — hybrid rule + ML decision logic
- `similarity.py` — original lightweight TF-IDF similarity utilities
- `recommendations.py` — improvement suggestions
- `database.py` — SQLite persistence
- `training_requirements.csv` — demonstration labeled ML data
- `sample_requirements.csv` — batch-analysis examples
- `test_ml_extensions.py`, `test_quality_engine.py`, `test_similarity.py` — automated tests

## How to explain the project

> ReqGuard AI uses machine learning and NLP to analyze the quality and risk of software requirements before development starts. It combines explainable rule-based checks with multiple machine-learning classifiers, quality-score regression, transformer-based semantic similarity, duplicate/conflict detection, and clustering. The aim is to identify unclear or risky requirements early and provide understandable recommendations for improvement.
