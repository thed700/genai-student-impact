<div align="center">

# 🎓 GenAI Student Impact Framework
### *Tutor vs. Cheat Code — Decoding How 50,000 Students Actually Use AI*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189AB4?style=for-the-badge&logo=data:image/png;base64,)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.20+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

**[🚀 Live Demo](https://your-app.streamlit.app)** &nbsp;·&nbsp;
**[📓 Notebook](notebooks/)** &nbsp;·&nbsp;
**[📊 Dataset](data/)** &nbsp;·&nbsp;
**[🐛 Issues](../../issues)**

<br/>

<img src="https://img.shields.io/badge/Records-50%2C000%20Students-6C63FF?style=flat-square" />
<img src="https://img.shields.io/badge/Models-XGBoost%20Classifier%20%2B%20Regressor-FF6584?style=flat-square" />
<img src="https://img.shields.io/badge/GPA%20Model%20R%C2%B2-0.9158-43D9AD?style=flat-square" />

</div>

---

## 📌 Live Demo

> **⚡ Try the interactive dashboard instantly:**
> [`https://your-app.streamlit.app`](https://genai-student-impact-vnbapw9uydqmjxddggu6vj.streamlit.app/)
> *Deploy your own in 2 minutes — see [How to Run](#-how-to-run-locally) below.*

---

## 🧭 Project Overview

> *"Every student using AI is running one of two programs: a personal tutor that amplifies their thinking, or a cheat code that quietly erodes it. The data knows which one."*

Generative AI arrived on campuses without a manual. Students adopted it rapidly, institutions scrambled to write policies, and nobody had the data to answer the questions that actually mattered:

- **At what point does AI assistance become AI dependency?**
- **Which students benefit most — and which are quietly burning out?**
- **Is there a measurable "safe zone" of AI usage that maximises performance without damaging wellbeing?**

This project answers all three with a **production-grade data science pipeline** built on **50,000 anonymised student records** spanning five major categories, five year levels, and three institutional policy regimes. The result is a two-model ML system and an interactive Streamlit application that gives any student or educator **personalised, evidence-based guidance** in seconds.

### Core Philosophy

The framework is built on a single principle: **the relationship between AI usage and outcomes is non-linear**. A little AI helps. More AI has diminishing returns. Beyond a threshold, it actively hurts both skill retention and mental health. Finding that threshold — and making it actionable — is the entire point.

---

## 🔍 Key Findings & Data Insights

### 📐 The Sweet Spot

Analysis of 50,000 records reveals a clear empirical "safe zone" for GenAI usage:

| Weekly AI Hours | Avg Skill Retention | High Burnout Rate | Verdict |
|:-:|:-:|:-:|:-:|
| 0 – 4 hrs | 75.9 / 100 | 9.8% | ✅ Safe — underpowered |
| **4 – 8 hrs** | **76.8 / 100** | **16.6%** | **🏆 Optimal zone** |
| **8 – 12 hrs** | **77.1 / 100** | **25.6%** | **⚠️ Peak retention, rising risk** |
| 12 – 16 hrs | 76.8 / 100 | 39.0% | ⚠️ Declining value |
| 16 – 20 hrs | 76.1 / 100 | 52.5% | 🔴 Burnout majority |
| 20 – 24 hrs | 75.2 / 100 | 62.0% | 🔴 Dangerous territory |
| 28 hrs+ | 65–67 / 100 | 83–93% | 💀 Severe harm zone |

> **The insight:** Skill retention *peaks* at 8–12 hrs/week, but so does risk. The **practical sweet spot is 4–8 hrs** — nearly identical retention with half the burnout rate. Beyond 20 hours, you're paying a 62%+ burnout tax for *lower* skill outcomes than a student using AI for 4 hours.

### 🏫 Major-Specific Optimal Profiles

Top-performing, low-burnout students (top 20% GPA, non-High burnout) share remarkably consistent patterns within each major:

| Major | Optimal AI Hours | Primary Use Case | Prompt Skill | Avg GPA |
|:--|:-:|:--|:-:|:-:|
| 🔬 **STEM** | 5.6 hrs/wk | Debugging / Troubleshooting | Advanced | 3.918 |
| 💼 **Business** | 5.1 hrs/wk | Ideation | Beginner | 3.907 |
| 📚 **Humanities** | 4.3 hrs/wk | Copywriting / Drafting | Beginner | 3.919 |
| 🏥 **Medical** | 4.4 hrs/wk | Summarising Reading | Beginner | 3.919 |
| 🎨 **Arts** | 4.4 hrs/wk | Copywriting / Drafting | Intermediate | 3.915 |

### 📊 Population-Level Stats

| Metric | Value |
|:--|:--|
| Total student records | 50,000 |
| Average weekly GenAI usage | 8.43 hrs |
| Average GPA change (pre → post) | **+0.203** |
| Students with High Burnout | 25.0% |
| Students with Low Burnout | 32.7% |
| Average Skill Retention Score | 75.8 / 100 |

---

## 🗂️ Repository Structure

```
genai-student-impact/
│
├── 📄 app.py                      # Streamlit application (3-tab dashboard)
├── 🤖 train_pipeline.py           # Phase 1: full ML training pipeline
├── 📋 requirements.txt            # Python dependencies
├── 📖 README.md                   # This file
│
├── 📁 data/
│   └── ai_student_impact_dataset.csv   # 50,000 student records (add via Git LFS)
│
├── 📁 models/                     # Serialised artefacts (joblib)
│   ├── burnout_model.pkl          # XGBoost Classifier → Burnout Risk Level
│   ├── gpa_model.pkl              # XGBoost Regressor  → Post-Semester GPA
│   ├── label_encoder.pkl          # LabelEncoder for Burnout classes
│   └── feature_pipeline.pkl       # Fitted ColumnTransformer (encode + scale)
│
├── 📁 notebooks/
│   └── 01_eda_and_modelling.ipynb # Exploratory analysis & model development
│
└── 📁 .streamlit/
    └── config.toml                # Dark-mode theme configuration
```

---

## 🧠 Machine Learning Architecture

The system is a **dual-model pipeline** trained on identical feature engineering:

```
Raw CSV Input (50,000 records)
         │
         ▼
┌─────────────────────────────────────────────┐
│          ColumnTransformer (fitted)          │
│                                             │
│  OrdinalEncoder ──► Year_of_Study           │
│                     Prompt_Engineering_Skill │
│                                             │
│  OneHotEncoder  ──► Major_Category          │
│                     Primary_Use_Case        │
│                     Institutional_Policy    │
│                                             │
│  StandardScaler ──► All numeric features    │
└─────────────────────────────────────────────┘
         │
    (23 features)
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌──────────┐
│ XGB   │  │ XGB      │
│Classif│  │Regressor │
│ier    │  │          │
└───────┘  └──────────┘
    │              │
    ▼              ▼
Burnout Risk   Post-GPA
(Low/Med/High) (1.0–4.0)
```

### Model 1 — Burnout Risk Classifier

| Parameter | Value |
|:--|:--|
| Algorithm | `XGBClassifier` |
| Objective | `multi:softprob` (3 classes) |
| Estimators | 500 trees |
| Learning rate | 0.05 |
| Max depth | 6 |
| Subsampling | 0.8 (rows) · 0.8 (cols) |
| Validation accuracy | **52.1%** *(baseline: 33.3%)* |
| Macro F1 | **0.52** |

> The classification task is intentionally hard — the "Medium" burnout class is a genuinely ambiguous psychological category. The model outperforms random chance by **+57%** and provides calibrated probability scores that are more useful than a hard label.

### Model 2 — GPA Regressor

| Parameter | Value |
|:--|:--|
| Algorithm | `XGBRegressor` |
| Objective | `reg:squarederror` |
| Estimators | 500 trees |
| Learning rate | 0.05 |
| **R² Score** | **0.9158** |
| **MAE** | **0.1132 GPA points** |

> The regression model is highly accurate — explaining **91.6% of variance** in post-semester GPA with average predictions within ±0.11 GPA points of actual outcomes.

### Feature Engineering

Encoding strategy was chosen to respect the semantics of each feature type:

- **Ordinal encoding** for ranked categoricals (e.g., `Freshman → 0 … Graduate → 4`) — preserves the meaningful rank signal that one-hot encoding destroys.
- **One-hot encoding** for nominal categoricals — no false ordinal relationships introduced.
- **Standard scaling** for all numeric features — ensures gradient descent stability and comparability across different-magnitude inputs.

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.11+
- pip or conda

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/thed700/genai-student-impact.git
cd genai-student-impact

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Place the dataset
#    Copy your CSV into the data/ folder:
#    data/ai_student_impact_dataset (1).csv

# 5. (Optional) Retrain the models
#    Skip this if you use the pre-trained .pkl files in /models
python train_pipeline.py

# 6. Launch the Streamlit app
streamlit run app.py
```

The app opens at **`http://localhost:8501`** automatically.

### Deploy to Streamlit Cloud (free)

1. Fork this repo to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your fork · branch `main` · file `app.py`.
4. Add your dataset via the **Secrets** panel or use Git LFS.
5. Click **Deploy** — live in ~90 seconds.

---

## 🛠️ Technologies Used

| Layer | Technology | Purpose |
|:--|:--|:--|
| **Language** | ![Python](https://img.shields.io/badge/-Python_3.11-3776AB?logo=python&logoColor=white&style=flat-square) | Core runtime |
| **Data** | ![Pandas](https://img.shields.io/badge/-Pandas_2.1-150458?logo=pandas&logoColor=white&style=flat-square) | Data wrangling & aggregation |
| **Numerics** | ![NumPy](https://img.shields.io/badge/-NumPy_1.26-013243?logo=numpy&logoColor=white&style=flat-square) | Array operations |
| **ML Framework** | ![scikit-learn](https://img.shields.io/badge/-scikit--learn_1.4-F7931E?logo=scikit-learn&logoColor=white&style=flat-square) | Preprocessing pipelines |
| **Boosting** | ![XGBoost](https://img.shields.io/badge/-XGBoost_2.0-189AB4?style=flat-square) | Classification & regression models |
| **Serialisation** | ![joblib](https://img.shields.io/badge/-joblib_1.3-4B8BBE?style=flat-square) | Model persistence (.pkl) |
| **Web App** | ![Streamlit](https://img.shields.io/badge/-Streamlit_1.35-FF4B4B?logo=streamlit&logoColor=white&style=flat-square) | Interactive dashboard |
| **Visualisation** | ![Plotly](https://img.shields.io/badge/-Plotly_5.20-3F4F75?logo=plotly&logoColor=white&style=flat-square) | Interactive charts |

---

## 📱 App Preview

### Tab 1 — Macro Insights Dashboard
Interactive KPI metrics, violin plots of AI hours by burnout level, a heatmap of burnout × exam anxiety, and the dual-axis "Sweet Spot" chart with an annotated optimal-usage band.

### Tab 2 — Personalised Recommendations
Filter 50,000 records by your Major + Year + Policy to see what the top 20% of performers in your exact cohort actually do differently — rendered as data-driven recommendation cards with comparison scatter plots.

### Tab 3 — ML Impact Predictor
13 input sliders and dropdowns feed both XGBoost models live. Returns colour-coded burnout risk (🔴🟡🟢) with confidence percentages, predicted GPA with cohort percentile histogram, and conditional actionable warnings when AI dependency exceeds safe thresholds.

---

## 🔬 Methodology Notes

**Train / Validation split:** 80/20 stratified on burnout class labels to prevent class imbalance skew.

**No data leakage:** `Post_Semester_GPA` is excluded from the burnout classifier feature set; `Burnout_Risk_Level` is excluded from the GPA regressor. The feature pipeline is fitted **only** on training data and applied to the validation set.

**Why XGBoost?** Tree-based boosting handles the mix of ordinal, nominal, and continuous features naturally, provides calibrated `predict_proba` outputs for the burnout risk display, and trains in under 90 seconds on 40,000 records — practical for a full retrain on any laptop.

---

## 🗺️ Roadmap

- [ ] SHAP explainability panel in Tab 3 (show *why* the model made its prediction)
- [ ] Time-series simulation: "What happens to my GPA if I increase AI hours by 2/week?"
- [ ] Clustering tab: unsupervised student archetype discovery (K-Means / UMAP)
- [ ] REST API wrapper (FastAPI) for LMS integration
- [ ] GPT-4o narrative summary of personalised recommendations

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome.

```bash
# Fork → clone → create branch
git checkout -b feature/your-feature-name

# Make changes, commit with conventional commits
git commit -m "feat: add SHAP explainability panel"

# Push and open a Pull Request
git push origin feature/your-feature-name
```

Please follow [Conventional Commits](https://www.conventionalcommits.org/) and include tests for any new pipeline logic.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 👤 Author

**Akmal Raxmatov**

[![GitHub](https://img.shields.io/badge/-@thed700-181717?style=flat-square&logo=github)](https://github.com/thed700)

---

<div align="center">

*Built with data, deployed with intent.*

⭐ **Star this repo** if the Sweet Spot finding changed how you think about AI study habits.

</div>
