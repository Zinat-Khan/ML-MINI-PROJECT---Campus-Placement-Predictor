# Campus Placement Predictor — Transparent ML Application

> **Version:** 2026.2.0-social (GitHub & LinkedIn Enhanced)  
> **Architecture:** FastAPI Backend + Vanilla JS Frontend + Scikit-learn ML Pipeline  

---

## 🎯 Project Overview

An interpretable machine learning system that estimates campus placement likelihood using historical student records, technical coding activity, GitHub contributions, and LinkedIn profile analysis. Every prediction includes:

- **Calibrated Probability** with 80% bootstrap confidence interval
- **Plain-language factor explanations** (top positive/negative influences)
- **What-If Simulator** for actionable counterfactual exploration
- **Responsible AI compliance** with fairness audits and mandatory disclosures

### Core Philosophy: *Advice, Not Verdict*

This tool helps students **prioritize preparation** — it does NOT rank, shortlist, or select candidates.

---

## 📂 Project Structure

```
miniproject/
├── README.md                          # This file
├── backend/
│   ├── requirements.txt               # Python dependencies
│   ├── train_model.py                 # Synthetic data generation, model training, fairness audit
│   ├── pipeline.py                    # Inference engine, explanations, what-if simulator
│   ├── main.py                        # FastAPI REST API server
│   ├── model.joblib                   # [Generated] Trained sklearn pipeline
│   ├── model_metadata.json            # [Generated] Model metrics & feature weights
│   └── synthetic_data.csv             # [Generated] Training dataset (600 records)
└── frontend/
    ├── index.html                     # Single-page application (5 tabs)
    ├── style.css                      # Dark glassmorphism design system
    └── app.js                         # Client ML engine + API integration
```

---

## 🚀 Quick Start

### Option A: Frontend Only (No Python Required)

Simply open `frontend/index.html` in any modern browser. The embedded client-side ML engine uses the **same logit coefficients** as the trained model, so predictions work without the backend.

### Option B: Full Stack (Python Backend + Frontend)

**Step 1: Install Python Dependencies**
```bash
cd miniproject/backend
pip install -r requirements.txt
```

**Step 2: Train the Model**
```bash
python train_model.py
```
This generates:
- `synthetic_data.csv` (600 historical placement records)
- `model.joblib` (calibrated logistic regression pipeline)
- `model_metadata.json` (metrics, feature weights, fairness audit)

**Step 3: Start the API Server**
```bash
python main.py
```
Server starts at `http://127.0.0.1:8000`. Swagger docs at `/docs`.

**Step 4: Open the Frontend**
Open `frontend/index.html` in a browser. It auto-detects the backend and switches from "Client ML Engine" to "FastAPI Connected" mode.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/v1/health` | Liveness check |
| `POST` | `/v1/predict` | Predict placement probability with factor explanations |
| `POST` | `/v1/what-if` | Simulate counterfactual changes (actionable features only) |
| `GET` | `/v1/model-info` | Model card metadata and evaluation metrics |
| `GET` | `/v1/insights/global` | Feature importance ranking |
| `GET` | `/v1/insights/cohort` | Cohort placement statistics by branch |
| `POST` | `/v1/consent` | Record consent preference |
| `DELETE` | `/v1/data` | Erase temporary inference data (GDPR-style) |

---

## 🧠 ML Pipeline Details

| Component | Implementation |
|:----------|:---------------|
| **Algorithm** | Regularized Logistic Regression (C=1.0) |
| **Calibration** | Platt Sigmoid Scaling (CalibratedClassifierCV, 5-fold) |
| **Features** | 20 numeric + 1 categorical (branch) |
| **Social Analysis** | GitHub repos/contributions/stars + LinkedIn connections/profile score |
| **Fairness Audit** | Post-hoc TPR/FPR gap analysis across gender subgroups |
| **Uncertainty** | 80% bootstrap confidence interval (simplified symmetric) |

### Release Gate Criteria

| Metric | Target | Status |
|:-------|:-------|:-------|
| ROC-AUC (5-fold CV) | ≥ 0.75 | ✅ Passed |
| Recall (at-risk class) | ≥ 0.70 | ✅ Passed |
| Expected Calibration Error | ≤ 0.08 | ✅ Passed |
| Max Gender TPR Gap | ≤ 0.10 | ✅ Passed |

---

## 🛡️ Responsible AI

- **Protected attributes excluded**: Gender, caste, religion never enter the model
- **Fairness audited**: Every training cycle checks TPR/FPR gaps across demographics
- **Mandatory disclosure**: Every UI screen renders limitation text about unmeasured soft skills
- **Data minimization**: No student data stored by default; opt-in only with DELETE endpoint

---

## 🖥️ Frontend Features (5 Tabs)

1. **Student Portal** — Input form with academic, technical, GitHub & LinkedIn fields → probability gauge + factor breakdown
2. **What-If Simulator** — Interactive sliders for actionable features with real-time delta calculation
3. **Cohort Insights** — Global feature importance bars + branch-wise placement statistics
4. **Model Card** — Official model documentation, metrics table, and stated limitations
5. **Audit & Admin** — Fairness audit results table + data erasure controls

---

## 📋 Stated Limitations

This system **cannot measure**:
- Confidence & Attitude (requires interview observation)
- Learning Ability & Adaptability (qualitative assessment)
- HR & Cultural Fit (interpersonal dynamics)
- Resume Quality (free-text NLP excluded in v1)

These are prominently disclosed in the UI and Model Card.

---

*Developed under Responsible AI Guidelines. Campus Placement Predictor © 2026.*
