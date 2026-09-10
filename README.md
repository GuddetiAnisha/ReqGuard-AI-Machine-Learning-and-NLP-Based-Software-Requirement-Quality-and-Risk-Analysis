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
- `data/training_requirements.csv` — demonstration labeled ML data
- `data/sample_requirements.csv` — batch-analysis examples
- `tests/` — automated tests

## How to explain the project

> ReqGuard AI uses machine learning and NLP to analyze the quality and risk of software requirements before development starts. It combines explainable rule-based checks with multiple machine-learning classifiers, quality-score regression, transformer-based semantic similarity, duplicate/conflict detection, and clustering. The aim is to identify unclear or risky requirements early and provide understandable recommendations for improvement.
