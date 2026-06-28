<<<<<<< HEAD
# 🏥 MedGuard AI
AI-powered Medical Diagnosis System

## 🔍 Features
- 🫁 Chest X-Ray Analysis (Pneumonia Detection)
- ❤️ Heart Disease Prediction
- 🩸 Diabetes Prediction

---

## 📥 Dataset Download

| Dataset | Link | Size |
|---------|------|------|
| 🫁 Chest X-Ray (Pneumonia) | [Download from Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | ~2 GB |

> ⚠️ Download dataset `data/` extract in folder

---

## 📌 Overview

MedGuard AI is a comprehensive healthcare AI platform designed to assist in early disease detection across three clinical domains. Each prediction system is paired with SHAP-based Explainable AI to provide transparent, clinically interpretable results that clinicians and patients can actually understand.

> ⚠️ **Disclaimer:** MedGuard AI is built for educational and research purposes. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## ✨ Features

| Module | Description |
|---|---|
| 🫀 **Heart Disease Prediction** | Classification model with SHAP waterfall plots for per-feature cardiac risk explanation |
| 🩸 **Diabetes Risk Assessment** | Identifies diabetes risk factors with interpretable feature contributions |
| 🫁 **Pneumonia Detection** | Image-based chest X-ray classification using deep learning |
| 🔬 **SHAP Explainability** | Transparent predictions across all models — no black boxes |
| 📄 **PDF Diagnostic Report** | Downloadable patient report with predictions and interpretations |
| 📊 **Visual Analytics Dashboard** | Model interpretation dashboards for each disease module |

---

## 🖼️ App Preview

> *(Add screenshots here once deployed. Drag & drop images into this file on GitHub.)*

```
![MedGuard Dashboard](screenshot-dashboard.png)
![SHAP Explanation](screenshot-shap.png)
```

---

## 🗂️ Project Structure

```
medguard-Ai/
│
├── app.py                        # Main Streamlit app — navigation & routing
│
├── modules/
│   ├── heart/
│   │   ├── heart_app.py          # Heart disease prediction UI
│   │   ├── heart_pipeline.pkl    # Trained heart disease model
│   │   └── heart_explainer.pkl   # SHAP explainer for heart model
│   │
│   ├── diabetes/
│   │   ├── diabetes_app.py       # Diabetes risk assessment UI
│   │   ├── diabetes_pipeline.pkl # Trained diabetes model
│   │   └── diabetes_explainer.pkl
│   │
│   └── pneumonia/
│       ├── pneumonia_app.py      # Chest X-ray upload and classification UI
│       └── pneumonia_model/      # Pneumonia image classification model
│
├── utils/
│   ├── report.py                 # PDF diagnostic report generation
│   └── shap_helpers.py          # Shared SHAP plotting utilities
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.9+

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Taimoorahmad1789/medguard-Ai.git
cd medguard-Ai

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🔬 How Each Module Works

### 🫀 Heart Disease
```
Clinical Inputs (age, BP, cholesterol, ECG...)
        ↓
Scikit-learn Classification Pipeline
        ↓
Cardiac Risk Score + SHAP Waterfall Plot
        ↓
Downloadable PDF Diagnostic Report
```

### 🩸 Diabetes
```
Patient Metrics (glucose, BMI, insulin, age...)
        ↓
Scikit-learn Classification Pipeline
        ↓
Diabetes Risk Level + SHAP Feature Contributions
```

### 🫁 Pneumonia (Chest X-Ray)
```
Uploaded Chest X-Ray Image
        ↓
Image Classification Model
        ↓
Normal / Pneumonia Prediction + Confidence Score
```

---

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **ML**: Scikit-learn, SHAP
- **Image AI**: (Deep learning framework for pneumonia module)
- **Data**: Pandas, NumPy
- **Frontend**: Streamlit
- **Visualisation**: Plotly, Matplotlib, Seaborn
- **Reporting**: FPDF / ReportLab (PDF generation)

---

## 🚧 Development Status

| Module | Status |
|---|---|
| 🫀 Heart Disease Prediction | ✅ Complete |
| 🩸 Diabetes Risk Assessment | ✅ Complete |
| 🫁 Pneumonia Detection | ✅ Complete |
| 📄 PDF Diagnostic Reports | ✅ Complete |
| 🌐 Streamlit Cloud Deployment | 🔄 In Progress |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Taimoor Ahmad** — Junior Machine Learning Engineer

[![Portfolio](https://img.shields.io/badge/Portfolio-6366f1?style=flat-square)](https://taimoorahmad1789.github.io/taimoor-portfolio)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat-square)](mailto:taimoor.ahmad.ai@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github)](https://github.com/Taimoorahmad1789)
