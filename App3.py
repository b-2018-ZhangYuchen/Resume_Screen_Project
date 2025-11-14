import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# Streamlit settings
# ==========================
st.set_page_config(page_title="Resume Matcher", layout="centered")
st.title("🤖 Resume Matcher (Negative Training Version)")

# ==========================
# Load Model
# ==========================
model = joblib.load("xgb_resume_with_negatives.pkl")
preprocessor = joblib.load("preprocessor_with_negatives.pkl")
bert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ==========================
# Load dataset to get job role mapping
# ==========================
@st.cache_data
def load_dataset():
    df = pd.read_csv("job_applicant_dataset.csv", encoding="Windows-1252")
    role_to_desc = df.groupby("Job Roles")["Job Description"].first().to_dict()
    return df, role_to_desc

df, role_to_desc = load_dataset()

# ==========================
# User Inputs
# ==========================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 65, 30)
    gender = st.selectbox("Gender", df["Gender"].unique())
    race = st.selectbox("Race", df["Race"].unique())
    ethnicity = st.selectbox("Ethnicity", df["Ethnicity"].unique())

with col2:
    job_role = st.selectbox("Job Role", sorted(role_to_desc.keys()))
    user_desc = st.text_area("Candidate Job Description",
                             "Experienced in data analysis and ML model building.")
    resume = st.text_area("Resume Text",
                          "Skilled in Python, data science, and visualization.")

standard_desc = role_to_desc[job_role]
st.info(f"📘 Standard Job Description for **{job_role}**:\n\n{standard_desc}")

# ==========================
# Prediction Section
# ==========================
if st.button("🔮 Predict Match Probability"):

    full_desc = user_desc + " " + standard_desc

    # === 1. BERT embeddings (correct shape) ===
    resume_emb = bert_model.encode([resume], convert_to_numpy=True)[0]   # shape: (384,)
    job_emb = bert_model.encode([full_desc], convert_to_numpy=True)[0]  # shape: (384,)

    # concat text features → becomes (768,)
    X_text = np.hstack([resume_emb, job_emb]).reshape(1, -1)
    # NOW shape is (1, 768)  <-- FIXED

    # === 2. semantic similarity ===
    sim = cosine_similarity([resume_emb], [job_emb])[0][0]

    # === 3. tabular features ===
    df_input = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Race": race,
        "Ethnicity": ethnicity,
        "Job Roles": job_role,
        "semantic_similarity": sim
    }])

    X_tab = preprocessor.transform(df_input).toarray()  # shape: (1, N)

    # === 4. final feature ===
    X_final = np.hstack([X_text, X_tab])   # both are 2D → SAFE

    st.write(f"Model expects features: {model.n_features_in_}")
    st.write(f"App computed features: {X_final.shape[1]}")

    # === 5. predict ===
    y_pred = model.predict(X_final)[0]
    y_proba = model.predict_proba(X_final)[0][1]

    if y_pred == 1:
        st.success(f"✅ Suitable (Match Probability: {y_proba:.2f})")
    else:
        st.error(f"❌ Not Suitable (Match Probability: {y_proba:.2f})")

    # plot
    fig, ax = plt.subplots()
    ax.bar(["Not Match", "Match"], model.predict_proba(X_final)[0], color=["red", "green"])
    st.pyplot(fig)
