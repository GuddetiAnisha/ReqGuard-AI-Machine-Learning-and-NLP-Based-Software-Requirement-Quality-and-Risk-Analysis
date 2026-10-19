# ReqGuard AI — Detailed Project Explanation

## 1. Introduction

ReqGuard AI is an independent software engineering and artificial intelligence project that examines the quality of requirements written in everyday language. It helps identify statements that may be vague, incomplete, difficult to test, repeated, or potentially inconsistent with other requirements.

Its purpose is to identify possible requirement problems **before** developers begin implementing the software. The project combines explainable rules, supervised machine learning, natural language processing (NLP), semantic similarity, and clustering. It provides results through an interactive Streamlit dashboard.

This is an independent demonstration and portfolio project. **It is not an Ericsson or Volvo project, and its results should not be presented as work performed for either company.** The included training data is synthetic, not a proprietary industrial dataset.

## 2. Why requirement quality matters

Software developers first need to understand what a system is expected to do. When requirements are unclear, different people may interpret them differently. This can lead to rework, misunderstandings, missing tests, and unnecessary cost.

For example, consider:

> The application should load quickly.

The word "quickly" does not provide a measurable condition. One person may think two seconds is fast, while another expects half a second.

A clearer statement would be:

> Under the specified test conditions, the application shall display the home page within two seconds for at least 95% of requests.

This is only an example of a better-specified requirement. The actual performance target must come from stakeholders, not from an AI-generated guess.

ReqGuard AI aims to highlight issues like these and suggest what information the author should clarify.

## 3. Main objective

The main goal is to build a system that:

- evaluates requirement clarity, completeness, testability, specificity, and ambiguity;
- predicts whether a requirement has LOW, MEDIUM, or HIGH risk;
- estimates a numerical requirement-quality score;
- combines transparent rule-based analysis with learned predictions;
- identifies semantically similar or potentially contradictory requirements;
- groups related requirements into clusters;
- visualizes and exports the results.

The project assesses the **quality of requirements**. It does not implement the described software, prove its safety, or automatically validate every stakeholder intention.

## 4. How the system works

The workflow can be understood in seven stages:

1. **Input:** A user enters one requirement or uploads a CSV containing many requirements.
2. **Rule analysis:** The system searches for known indicators of vagueness and missing measurable detail, producing interpretable quality scores and issues.
3. **Text representation:** TF-IDF transforms requirement text into numeric features that machine-learning algorithms can process.
4. **Machine learning:** A trained classifier predicts the risk category, and a separate regressor predicts a quality score.
5. **Hybrid assessment:** The application combines the rule score and the ML quality prediction and determines the final risk.
6. **Semantic analysis:** Optional Sentence-BERT embeddings or TF-IDF support similarity, candidate duplicate detection, conflict checks, and grouping.
7. **Output:** Streamlit shows scores, risk levels, charts, explanations, recommendations, and exportable results.

Each part solves a different problem. Rules help explain common writing issues, learned models identify patterns from labelled examples, and semantic methods examine relationships between different statements.

## 5. Rule-based quality analysis

The file `quality_engine.py` produces an explainable assessment. Its output covers five dimensions:

| Dimension | Simple meaning |
| --- | --- |
| Clarity | Is the statement understandable rather than vague? |
| Completeness | Does it contain enough relevant information? |
| Testability | Could someone verify whether the requirement was met? |
| Specificity | Is the expected behavior sufficiently precise? |
| Ambiguity quality | Does the wording avoid uncertain or subjective expressions? |

These are heuristic software-produced scores, not certified human review ratings.

The file `ambiguity.py` supports the identification of ambiguous expressions. The recommendation module uses identified issues to give the user suggestions or a possible rewrite. Suggested rewrites are starting points, not authoritative replacements for stakeholder-approved requirements.

## 6. Machine-learning risk classification

This is a genuine supervised machine-learning component: the project fits models to example requirements with known risk labels.

The labelled categories are:

- **LOW:** The example is relatively clear and measurable.
- **MEDIUM:** Some important details could be improved.
- **HIGH:** The example contains more serious clarity or specification concerns.

These labels describe **requirement-quality risk** in the prototype. They are not a measured probability of project failure or an industrial safety-risk rating.

The file `ml_model.py` uses scikit-learn pipelines with TF-IDF features and compares four algorithms.

### Logistic Regression

A statistical classification method that learns relationships between text features and risk labels. Despite its name, it is used here for classification, not continuous-value regression.

### Random Forest Classifier

An ensemble of decision trees that makes predictions using patterns learned from labelled examples. It can capture more complex relationships between features.

### Linear Support Vector Machine (Linear SVM)

A classifier that learns boundaries separating the risk classes in a high-dimensional feature space. TF-IDF text representations often produce many features, making linear models useful comparison baselines.

### Multinomial Naive Bayes

A probabilistic text-classification baseline that works with nonnegative word-frequency-style features. It makes simplifying assumptions about feature independence.

All four classifiers are trained and evaluated, and the implementation selects a model primarily by **weighted F1-score**, with accuracy as a tie-breaker.

## 7. Training and testing

The included `training_requirements.csv` is described in the repository as a synthetic demonstration dataset of 150 labelled requirements.

The code uses `train_test_split` with a 25% test fraction, random seed 42, and stratification of risk labels. That gives 112 training examples and 38 held-out examples for the documented classification experiment.

The training portion teaches the algorithm patterns in labelled text. The held-out portion is used to estimate performance on examples not included in model fitting.

**Important:** A held-out split of a small, intentionally prepared synthetic dataset is not proof of generalization to real projects. Independent documents, stronger annotation procedures, and more diverse examples would be needed for a research-grade claim.

## 8. Classification evaluation

The project uses several metrics:

| Metric | What it tells us |
| --- | --- |
| Accuracy | The proportion of test examples classified correctly |
| Precision | How often predicted examples of a category are actually in that category |
| Recall | How many examples of a category the model successfully identifies |
| F1-score | A combined measure balancing precision and recall |
| Confusion matrix | Which classes the model predicts correctly and where mistakes occur |

The README records the following weighted metrics from the bundled demonstration benchmark:

| Model | Accuracy | Weighted F1 |
| --- | ---: | ---: |
| Random Forest | 0.9737 | 0.9737 |
| Linear SVM | 0.9211 | 0.9222 |
| Logistic Regression | 0.8684 | 0.8649 |
| Naive Bayes | 0.8158 | 0.8090 |

Random Forest performed best on that documented synthetic split. These are **previously reported local validation results**, not measurements newly reproduced while writing this explanation.

## 9. Machine-learning quality regression

Classification predicts a category such as HIGH risk. Regression predicts a numerical value.

ReqGuard AI trains a separate **Random Forest Regressor** to estimate requirement quality on a scale from 0 to 100. The pipeline uses TF-IDF text features and the labelled `quality_score` column.

The README reports these results on its synthetic test split:

| Metric | Reported result |
| --- | ---: |
| Mean Absolute Error (MAE) | 4.57 |
| Root Mean Squared Error (RMSE) | 5.64 |
| R-squared (R²) | 0.901 |

MAE describes the average absolute difference between predicted and reference scores. RMSE penalizes larger errors more strongly. R² describes how much variation in the test target is accounted for relative to predicting its mean.

These scores reflect the particular synthetic examples and their assigned quality labels, not an independently verified universal measure of requirement quality.

## 10. Hybrid decision system

The file `hybrid.py` combines the interpretable rule score and the ML regressor score:

```text
Final quality = 0.60 × Rule quality + 0.40 × ML-predicted quality
```

For example, if the rule score is 70 and the ML score is 80:

```text
Final quality = 0.60 × 70 + 0.40 × 80 = 74
```

The implementation turns the final quality into a risk signal:

- 80 or above: LOW quality-related risk
- 60 to below 80: MEDIUM
- Below 60: HIGH

The final risk is the **most conservative (highest)** risk category among the rule-based risk, the quality-derived risk, and the ML classifier risk when available.

If no trained regressor is available, the system uses the rule score as the final quality value. This behavior makes the output more robust to missing model artifacts, but the usefulness of the hybrid weights has not been independently established by a large external study.

## 11. Traditional NLP: TF-IDF and cosine similarity

TF-IDF stands for **Term Frequency–Inverse Document Frequency**. In simple terms, it gives importance to words that help distinguish one requirement from another.

Cosine similarity compares two numerical text representations. A higher similarity can suggest that the requirements discuss similar topics.

For example:

> The user shall reset a forgotten password through email.

> The application shall provide email-based password recovery.

Although the wording differs, the meaning is similar. Plain TF-IDF may detect overlapping terms, but semantic embeddings can better represent some paraphrases.

Similarity does **not** by itself prove two requirements are duplicates. Their scope, actors, conditions, and exact obligations may differ.

## 12. Sentence-BERT semantic embeddings

The file `semantic_analysis.py` supports the Sentence-BERT model `all-MiniLM-L6-v2` when the optional model is available.

Sentence-BERT converts a sentence into an embedding: a vector of numbers intended to capture aspects of its meaning. Similar meanings tend to have similar representations.

ReqGuard AI uses this capability for meaning-based requirement comparisons. If the transformer model cannot be loaded, the repository describes a fallback to TF-IDF-based analysis.

The first use of Sentence-BERT may download model files. Model availability and download requirements depend on the local environment.

## 13. Duplicate and conflict detection

Two separate kinds of review are supported:

**Potential duplicates:** Two requirements may express much the same obligation using different words. Similarity scoring produces candidates for human review, not automatic deletion.

**Potential conflicts:** Two requirements may discuss the same feature but express incompatible conditions, such as contradictory negation or numeric limits. The project uses similarity together with simple textual contradiction signals to flag candidates.

For example:

> The service shall retain audit records for 90 days.

> The service shall delete all audit records after 30 days.

These statements may be inconsistent if they apply to the same records and conditions. A reviewer must still check context before declaring a genuine conflict.

## 14. Unsupervised learning: K-Means clustering

The `clustering.py` module groups related requirements.

Unlike supervised risk classification, K-Means does not need a risk label for every requirement. It groups text representations into a user-selected number of clusters based on feature similarity.

A cluster might contain requirements about authentication, another about performance, and another about reporting. The algorithm does not necessarily assign those descriptive names; people interpret the resulting groups.

Clustering is useful for organizing a requirement collection, but it does not prove that each cluster represents one correctly separated system feature.

## 15. Streamlit dashboard

The application's `app.py` defines five main tabs:

| Tab | What the user can do |
| --- | --- |
| Single Requirement | Enter text and see rule quality, ML quality, final risk, issues, and suggestions |
| Batch Analysis | Analyze multiple CSV rows, group requirements, and inspect duplicate/conflict candidates |
| Semantic NLP | Compare text similarity using traditional and transformer-based methods |
| ML Lab | Explore model training, evaluation, and comparison |
| History | Inspect saved analysis records |

The application uses pandas and Plotly to display data, metrics, and charts. Batch analysis supports CSV export.

## 16. SQLite history

The `database.py` module provides local SQLite persistence for analysis history.

This allows the application to save analysis records for later inspection without requiring a separate database server. It should not be treated as a multi-user production deployment with access controls.

## 17. Project file structure

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit application |
| `quality_engine.py` | Explainable requirement quality scoring |
| `ambiguity.py` | Detection of ambiguous terms |
| `ml_model.py` | Risk classification, model comparison, quality regression, model saving/loading |
| `hybrid.py` | Combining rule and ML predictions |
| `semantic_analysis.py` | TF-IDF/transformer semantic comparison, duplicate and conflict candidates |
| `similarity.py` | Lightweight similarity utilities |
| `clustering.py` | K-Means grouping |
| `recommendations.py` | Improvement suggestions |
| `database.py` | SQLite history |
| `training_requirements.csv` | Bundled synthetic labelled examples |
| `sample_requirements.csv` | Bundled examples for batch use |
| `test_ml_extensions.py`, `test_quality_engine.py`, `test_similarity.py` | Automated tests |

### Note about file paths in this repository

At the time this document was prepared, the repository root includes `training_requirements.csv` and `sample_requirements.csv`, while `app.py` looks for its automatic datasets under a `data/` directory.

If automatic sample loading or training does not work after cloning, create a `data` folder and copy the two CSV files into it. This is a setup detail, not a reason to claim that automatic training definitely succeeded on a fresh clone.

## 18. Run in VS Code on Windows

Install Python, Git, and Visual Studio Code. Open a PowerShell terminal:

```powershell
git clone https://github.com/GuddetiAnisha/ReqGuard-AI-Machine-Learning-and-NLP-Based-Software-Requirement-Quality-and-Risk-Analysis.git
cd ReqGuard-AI-Machine-Learning-and-NLP-Based-Software-Requirement-Quality-and-Risk-Analysis
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For the CSV paths expected by the application, run:

```powershell
New-Item -ItemType Directory -Force data
Copy-Item training_requirements.csv data\training_requirements.csv
Copy-Item sample_requirements.csv data\sample_requirements.csv
```

Then start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://localhost:8501**. Use the **Single Requirement** tab to enter an example such as `The application should respond quickly.` and click **Analyze Requirement**.

To execute automated tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The README previously reports eight passing tests in its local verification. Re-run the current tests on your own machine rather than assuming that result is guaranteed in a different environment.

Installing `sentence-transformers` may also download large supporting packages such as PyTorch. If that installation is difficult, the README describes a lightweight TF-IDF fallback installation.

## 19. Limitations and responsible interpretation

The most important limitations are:

- The included labelled examples are synthetic and relatively small.
- Model performance on those examples does not establish real-world accuracy.
- Predicted confidence displays should not automatically be treated as calibrated probabilities.
- Heuristic rules can miss ambiguous language or incorrectly flag clear statements.
- Text similarity is not the same as logical equivalence.
- A possible conflict should be reviewed in its system context.
- Suggested rewrites must be reviewed by stakeholders before becoming requirements.
- There is no claim of industrial deployment or validation by an external company.

The system is best understood as a prototype that supports human requirement review, not as a tool that can independently approve a software specification.

## 20. Future improvements

Potential next steps include collecting a larger independently annotated requirement dataset, validating on documents from unseen projects, adding multiple human reviewers, measuring inter-annotator agreement, calibrating prediction confidence, and separating duplicate detection from formal contradiction checking.

It would also be useful to compare sentence embedding models, study changes in performance with different risk thresholds, quantify false alarms, and measure whether the tool actually reduces review time in controlled user studies.

## 21. Conclusion

ReqGuard AI combines multiple artificial intelligence and software engineering methods to identify possible problems in written requirements. Its strongest feature is that it connects interpretable quality rules, ML-based risk predictions, numerical quality regression, and semantic analysis in one application.

The project demonstrates supervised learning, unsupervised learning, NLP, hybrid decision logic, evaluation metrics, and interactive visualization in a realistic software requirements engineering use case. It remains an independent academic/portfolio-style prototype based on synthetic demonstration data.

For exact implementation details, see [README.md](README.md) and the source files in this repository.
