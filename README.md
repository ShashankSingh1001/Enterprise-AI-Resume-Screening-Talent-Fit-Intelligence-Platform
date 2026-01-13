# 🏢 Enterprise AI Resume Screening & Talent Fit Intelligence Platform

An enterprise-grade Responsible AI hiring platform that automates resume screening, job description (JD) matching, candidate ranking, explainable decisioning, and fairness auditing using NLP, Machine Learning and MLOps pipelines.

---

## 📌 Problem Statement

Large enterprises receive thousands of resumes for each open role.
Manual screening is slow, biased, inconsistent and expensive.
This project builds a **deployable, explainable and ethical AI hiring platform** that automatically parses resumes, matches them semantically with job descriptions, predicts interview shortlisting probability, explains model decisions and audits hiring bias.

---

## 🚀 Key Features

* Resume parsing using NLP (skills, education, experience, projects)
* Semantic JD–Resume matching using Sentence-BERT
* ML-based role-fit prediction (XGBoost / LightGBM)
* Explainable AI using SHAP & LIME
* Bias & fairness audit using Fairlearn
* MLflow-based MLOps & model registry
* Dataset versioning with DVC
* FastAPI microservices
* Streamlit HR dashboard
* Exportable HR hiring reports (CSV/PDF)
* JWT-based role authentication

---

## 🧠 Technology Stack

Python, spaCy, Sentence-BERT, XGBoost, LightGBM, SHAP, LIME, Fairlearn, MLflow, DVC, FastAPI, Streamlit, PostgreSQL, Docker, GitHub Actions

---

## 🏗 System Architecture

```
Client (HR Dashboard)
        │
   API Gateway (FastAPI)
        │
Resume NLP ─ JD NLP Services
        │
Similarity & Feature Store
        │
AutoML Training Pipeline
        │
Model Registry (MLflow)
        │
Explainability + Bias Audit
        │
PostgreSQL / File Store
        │
Monitoring + Logs
```

---

## 📊 Dataset

* Kaggle Resume Dataset
* IT Job Description Dataset
* Synthetic hiring labels (enterprise simulation)

Final dataset schema:

```
resume_text, jd_text, similarity, exp_years, edu_level, skill_match_ratio, selected
```

---

## 🧪 ML Modeling

* SBERT semantic similarity scoring
* XGBoost / LightGBM ensemble classifier
* SHAP & LIME explainability
* Fairlearn bias metrics (gender, college tier, experience)

---

## 📁 Repository Structure

```
resume-ai/
 ├── api/
 ├── bias_audit/
 ├── dashboard/
 ├── data/
 ├── explainability/
 ├── features/
 ├── mlops/
 ├── models/
 ├── notebooks/
 └── README.md
```

---

## ⚙ Setup Instructions

### 1. Clone Repository

```
git clone https://github.com/your-username/resume-ai.git
cd resume-ai
```

### 2. Create Virtual Environment

```
python -m venv venv
source venv/bin/activate  (Windows: venv\Scripts\activate)
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Train Model

```
python mlops/train.py
```

### 5. Start API

```
uvicorn api.main:app --reload
```

### 6. Run Dashboard

```
streamlit run dashboard/app.py
```

---

## 📈 Output

* Ranked candidate list
* Fit probability score
* Explainability graphs
* Bias audit reports
* Downloadable hiring reports

---

## 📝 Resume Bullet

> Built an enterprise-grade AI Resume Screening & Talent Fit Prediction platform using SBERT, XGBoost, SHAP, Fairlearn, FastAPI and Streamlit, with MLflow-based MLOps and bias-aware Responsible AI pipelines.

---

## 🎯 Why This Project Matters

This project demonstrates **real enterprise AI engineering**, covering NLP, ML, Responsible AI, MLOps, and deployment — aligned with Infosys HR Tech and Digital Transformation verticals.


