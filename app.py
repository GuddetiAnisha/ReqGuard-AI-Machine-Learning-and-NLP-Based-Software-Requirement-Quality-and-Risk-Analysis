from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from database import init_db, save_analysis, load_recent
from quality_engine import score_requirement
from recommendations import generate_recommendations, suggest_rewrite
from ml_model import (
    load_model, predict_risk, train_and_compare_models,
    train_quality_regressor, load_quality_regressor,
)
from semantic_analysis import (
    find_semantic_duplicates, find_semantic_conflicts,
    compare_tfidf_vs_transformer, embedding_backend,
)
from clustering import cluster_requirements
from hybrid import hybrid_assessment

BASE_DIR = Path(__file__).parent
SAMPLE_DATA = BASE_DIR / "data" / "sample_requirements.csv"
TRAINING_DATA = BASE_DIR / "data" / "training_requirements.csv"

st.set_page_config(page_title="ReqGuard AI", page_icon="🛡️", layout="wide")
init_db()

@st.cache_resource
def get_classifier():
    model = load_model()
    if model is None and TRAINING_DATA.exists():
        try:
            model, _, _, _, _ = train_and_compare_models(TRAINING_DATA)
        except Exception:
            model = None
    return model

@st.cache_resource
def get_regressor():
    model = load_quality_regressor()
    if model is None and TRAINING_DATA.exists():
        try:
            model, _ = train_quality_regressor(TRAINING_DATA)
        except Exception:
            model = None
    return model

classifier = get_classifier()
regressor = get_regressor()

st.title("🛡️ ReqGuard AI")
st.caption("Machine Learning and NLP-Based Software Requirement Quality and Risk Analysis")

with st.sidebar:
    st.header("AI / ML Modules")
    st.write("• Explainable rule-based quality analysis")
    st.write("• TF-IDF text representation")
    st.write("• Logistic Regression")
    st.write("• Random Forest")
    st.write("• Linear SVM")
    st.write("• Naive Bayes")
    st.write("• Random Forest quality regression")
    st.write("• Sentence-BERT semantic embeddings")
    st.write("• Cosine-similarity duplicate detection")
    st.write("• K-Means requirement clustering")
    st.write("• Semantic conflict detection")
    st.divider()
    st.caption(f"Semantic backend: {embedding_backend()}")
    st.info("ReqGuard analyzes software requirements. It does not evaluate AI-generated test cases.")

single_tab, batch_tab, semantic_tab, model_tab, history_tab = st.tabs([
    "Single Requirement", "Batch Analysis", "Semantic NLP", "ML Lab", "History"
])

with single_tab:
    st.subheader("Hybrid requirement analysis")
    sample = "The application should load quickly and provide users with appropriate results."
    text = st.text_area("Software requirement", value=sample, height=140)

    if st.button("Analyze Requirement", type="primary", use_container_width=True):
        result = hybrid_assessment(text, classifier, regressor)
        rule = result["rule_result"]
        save_analysis(text, rule)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rule Quality", f"{rule.overall:.1f}/100")
        c2.metric("ML Quality", "N/A" if result["ml_quality_score"] is None else f"{result['ml_quality_score']:.1f}/100")
        c3.metric("Final Hybrid Quality", f"{result['final_quality_score']:.1f}/100")
        c4.metric("Final Risk", result["final_risk"])

        c5, c6 = st.columns(2)
        c5.metric("Rule-Based Risk", rule.risk)
        c6.metric("ML Risk Prediction", result["ml_risk"] or "Model unavailable")

        score_df = pd.DataFrame({
            "Metric": ["Clarity", "Completeness", "Testability", "Specificity", "Ambiguity quality"],
            "Score": [rule.clarity, rule.completeness, rule.testability, rule.specificity, rule.ambiguity],
        })
        st.plotly_chart(px.bar(score_df, x="Metric", y="Score", range_y=[0,100], title="Explainable Requirement Quality Breakdown"), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.markdown("### Issues detected")
            if rule.issues:
                for issue in rule.issues:
                    st.warning(issue)
            else:
                st.success("No major rule-based quality issues detected.")
        with right:
            st.markdown("### Recommendations")
            for rec in generate_recommendations(text, rule):
                st.info(rec)
            st.markdown("### Suggested rewrite")
            st.code(suggest_rewrite(text), language=None)

        probs = result["ml_risk_probabilities"]
        if probs:
            prob_df = pd.DataFrame({"Risk": list(probs), "Probability": list(probs.values())})
            st.plotly_chart(px.bar(prob_df, x="Risk", y="Probability", range_y=[0,1], title="ML Risk Confidence"), use_container_width=True)

        st.caption("Hybrid quality = 60% explainable rule score + 40% ML-predicted quality score. Final risk uses the most conservative signal from rule score, ML risk, and hybrid quality.")

with batch_tab:
    st.subheader("Batch quality analysis + clustering")
    st.write("Upload a CSV containing a `requirement` column or use the included sample file.")
    uploaded = st.file_uploader("Upload requirements CSV", type=["csv"], key="batch_upload")
    use_sample = st.checkbox("Use included sample requirements", value=uploaded is None)
    source_df = pd.read_csv(uploaded) if uploaded is not None else (pd.read_csv(SAMPLE_DATA) if use_sample and SAMPLE_DATA.exists() else None)

    if source_df is not None:
        if "requirement" not in source_df.columns:
            st.error("CSV must contain a `requirement` column.")
        else:
            st.dataframe(source_df, use_container_width=True)
            cluster_count = st.slider("Number of clusters", 2, min(8, max(2, len(source_df))), min(3, max(2, len(source_df))))
            if st.button("Run Batch ML/NLP Analysis", use_container_width=True):
                texts = source_df["requirement"].fillna("").astype(str).tolist()
                labels, cluster_backend = cluster_requirements(texts, cluster_count, prefer_transformer=True)
                rows=[]
                for idx, req in enumerate(texts):
                    h = hybrid_assessment(req, classifier, regressor)
                    r = h["rule_result"]
                    rows.append({
                        "ID": source_df.iloc[idx].get("id", f"R{idx+1:03d}"),
                        "Requirement": req,
                        "Rule Quality": r.overall,
                        "ML Quality": h["ml_quality_score"],
                        "Hybrid Quality": h["final_quality_score"],
                        "Rule Risk": r.risk,
                        "ML Risk": h["ml_risk"] or "N/A",
                        "Final Risk": h["final_risk"],
                        "Cluster": int(labels[idx]),
                        "Issues": len(r.issues),
                    })
                results_df = pd.DataFrame(rows)
                st.session_state["batch_results_v2"] = results_df
                st.dataframe(results_df, use_container_width=True)
                st.caption(f"Clustering backend: {cluster_backend}")

                a,b = st.columns(2)
                with a:
                    st.plotly_chart(px.histogram(results_df, x="Final Risk", title="Final Risk Distribution"), use_container_width=True)
                with b:
                    st.plotly_chart(px.box(results_df, x="Cluster", y="Hybrid Quality", title="Quality by Requirement Cluster"), use_container_width=True)

                duplicates, dup_backend = find_semantic_duplicates(texts, threshold=0.72, prefer_transformer=True)
                conflicts, conflict_backend = find_semantic_conflicts(texts, prefer_transformer=True)

                st.markdown("### Semantic duplicate candidates")
                st.caption(f"Backend: {dup_backend}")
                if duplicates:
                    st.dataframe(pd.DataFrame([{
                        "Requirement A": rows[i]["ID"], "Requirement B": rows[j]["ID"], "Semantic Similarity %": sim
                    } for i,j,sim in duplicates]), use_container_width=True)
                else:
                    st.success("No semantic duplicates detected at the current threshold.")

                st.markdown("### Potential semantic conflicts")
                st.caption(f"Backend: {conflict_backend}")
                if conflicts:
                    st.dataframe(pd.DataFrame([{
                        "Requirement A": rows[i]["ID"], "Requirement B": rows[j]["ID"],
                        "Semantic Similarity %": sim, "Reason": reason
                    } for i,j,sim,reason in conflicts]), use_container_width=True)
                else:
                    st.success("No potential semantic conflicts detected.")

    if "batch_results_v2" in st.session_state:
        data = st.session_state["batch_results_v2"].to_csv(index=False).encode("utf-8")
        st.download_button("Download ML/NLP Results", data, "reqguard_ml_analysis.csv", "text/csv", use_container_width=True)

with semantic_tab:
    st.subheader("Traditional NLP vs Transformer semantics")
    a = st.text_area("Requirement A", "Registered users shall recover forgotten passwords using email.")
    b = st.text_area("Requirement B", "A user can reset a forgotten password through the registered email address.")
    if st.button("Compare Semantic Similarity", use_container_width=True):
        scores = compare_tfidf_vs_transformer(a, b)
        display = pd.DataFrame({"Method": list(scores), "Similarity": [scores[k] for k in scores]})
        st.dataframe(display, use_container_width=True)
        valid = display.dropna()
        if not valid.empty:
            st.plotly_chart(px.bar(valid, x="Method", y="Similarity", range_y=[0,1], title="TF-IDF vs Sentence-BERT Similarity"), use_container_width=True)
        if scores.get("Sentence-BERT") is None:
            st.warning("Sentence-BERT is unavailable in this environment, so only TF-IDF was computed. Install sentence-transformers and allow the first model download to enable transformer embeddings.")

with model_tab:
    st.subheader("ML training and model comparison")
    st.write("The built-in labeled dataset is a demonstration dataset. For thesis-level claims, replace or extend it with a larger manually labeled requirements dataset.")
    if st.button("Train Classifiers and Quality Regressor", type="primary", use_container_width=True):
        try:
            best_model, best_name, results, train_n, test_n = train_and_compare_models(TRAINING_DATA)
            reg_model, reg_metrics = train_quality_regressor(TRAINING_DATA)
            st.cache_resource.clear()
            st.success(f"Training complete. Best risk classifier: {best_name}")
            comparison = pd.DataFrame([{
                "Model": name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
            } for name,m in results.items()]).sort_values("F1", ascending=False)
            st.dataframe(comparison, use_container_width=True)
            st.plotly_chart(px.bar(comparison, x="Model", y=["Accuracy","Precision","Recall","F1"], barmode="group", range_y=[0,1], title="Classifier Performance Comparison"), use_container_width=True)

            chosen = results[best_name]
            st.markdown(f"### Confusion Matrix — {best_name}")
            cm = pd.DataFrame(chosen["confusion_matrix"], index=["LOW","MEDIUM","HIGH"], columns=["LOW","MEDIUM","HIGH"])
            st.dataframe(cm, use_container_width=True)
            st.text(chosen["classification_report"])

            st.markdown("### Quality Score Regression")
            q1,q2,q3 = st.columns(3)
            q1.metric("MAE", f"{reg_metrics['mae']:.2f}")
            q2.metric("RMSE", f"{reg_metrics['rmse']:.2f}")
            q3.metric("R²", f"{reg_metrics['r2']:.3f}")
            st.caption(f"Classifier split: {train_n} train / {test_n} test. Regressor split: {reg_metrics['train_size']} train / {reg_metrics['test_size']} test.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

with history_tab:
    st.subheader("Saved rule-analysis history")
    records = load_recent(100)
    if records:
        hist = pd.DataFrame(records)
        st.dataframe(hist, use_container_width=True)
        st.plotly_chart(px.line(hist.sort_values("id"), x="id", y="overall", title="Saved Requirement Quality"), use_container_width=True)
    else:
        st.info("No saved analyses yet.")
