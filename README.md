# 🩺 Breast Cancer Prediction App

This is a web app built with **Streamlit** that predicts whether a breast tumor is **Benign (non-cancerous)** or **Malignant (cancerous)** using a **Logistic Regression** model trained on the Breast Cancer Wisconsin dataset.

---
## link of app

## 🚀 Features

- 🔮 Predicts cancer diagnosis based on top 10 medical features
- 📊 Shows prediction confidence
- ✅ Simple UI with helpful feature explanations
- ⚙️ Built with scikit-learn and Streamlit

---

## 📦 Files in This Repo

| File                  | Description                                      |
|-----------------------|--------------------------------------------------|
| `app.py`              | Main Streamlit app file                          |
| `logistic_model.pkl`  | Trained logistic regression model (optional)     |
| `scaler.pkl`          | Fitted StandardScaler for input features         |
| `requirements.txt`    | Python dependencies                              |
| `README.md`           | You’re reading it 😉                              |

---

## 🧪 Input Features Used (Top 10)

- `radius_mean`
- `perimeter_mean`
- `area_mean`
- `concave points_mean`
- `concavity_mean`
- `compactness_mean`
- `radius_worst`
- `perimeter_worst`
- `area_worst`
- `concave points_worst`

Each feature represents a measurement extracted from cell nuclei images in breast mass biopsies.

---

## 🛠️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-streamlit.git
cd breast-cancer-streamlit
