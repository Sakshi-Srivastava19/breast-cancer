import streamlit as st
import numpy as np
import joblib

# ----------------------------
# ⚙️ App Configuration
# ----------------------------
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Prediction App")
st.markdown("Enter tumor characteristics below to predict whether it is **Benign (0)** or **Malignant (1)**.")

# ----------------------------
# 🔌 Load Model and Scaler
# ----------------------------
try:
    model = joblib.load('logistic_model.pkl')
    scaler = joblib.load('scaler.pkl')
    st.success("✅ Model and Scaler loaded successfully.")
except Exception as e:
    st.error("🚨 Failed to load model or scaler. Please ensure the `.pkl` files are in the root directory.")
    st.stop()

# ----------------------------
# 📘 Sidebar Info
# ----------------------------
with st.sidebar.expander("📘 Feature Descriptions", expanded=False):
    st.markdown("""
**Feature explanations:**
- **radius**: Mean distance from center to perimeter
- **texture**: Standard deviation of pixel intensities
- **perimeter**: Length around the tumor
- **area**: Total area of the tumor
- **smoothness**: Consistency of the shape boundary
- **compactness**: Perimeter² / area - 1
- **concavity**: Depth of inward curves
- **concave points**: Number of inward points
- **symmetry**: How symmetrical the tumor is
- **fractal dimension**: Shape complexity

🔹 Suffixes:
- `_mean`: Average across cells
- `_se`: Standard error
- `_worst`: Worst (largest) measurement
""")

st.sidebar.title("⚙️ About")
st.sidebar.info("Model: Logistic Regression\n\nPrediction: `0 = Benign`, `1 = Malignant`\n\nDataset: Breast Cancer Wisconsin")

# ----------------------------
# 🧬 Feature List
# ----------------------------
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# ----------------------------
# 🧾 Input Form
# ----------------------------
st.subheader("🧬 Input Tumor Features")
input_data = []
cols = st.columns(3)

for i, feature in enumerate(features):
    with cols[i % 3]:
        val = st.number_input(f"{feature}", min_value=0.0, value=1.0, format="%.4f")
        input_data.append(val)
    if i % 10 == 9:
        st.markdown("---")

# ----------------------------
# 🔮 Prediction
# ----------------------------
if st.button("🔍 Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 0:
        st.success("🟢 **Prediction: Benign (Non-cancerous)**")
    else:
        st.error("🔴 **Prediction: Malignant (Cancerous)**")
    
    st.write(f"📊 **Confidence: `{probability * 100:.2f}%`**")

# ----------------------------
# 📌 Footer
# ----------------------------
st.markdown("---")
st.markdown("🔬 *This app is built for educational purposes as part of Skillfied Mentor Task 1.*")
