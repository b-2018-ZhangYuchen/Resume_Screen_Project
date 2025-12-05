# 📌 Resume Matcher (BERT + XGBoost) — Streamlit App

This project provides an interactive **AI-powered resume–job matching system** built with:

* **Sentence-BERT** for semantic embeddings
* **XGBoost** for classification
* **Streamlit** for a user-friendly interface

The app predicts **match probability** between a candidate’s resume and a job role, and provides fairness visualizations across demographic groups.

---

# 🚀 How to Run the Streamlit App

### **1. Install Dependencies**

### You can visit https://app-resume-screen.streamlit.app/ to get access to the application or...

Make sure your environment includes Python 3.8–3.10.

```bash
pip install -r requirements.txt
```

If running locally and you want to avoid widget warnings:

```bash
pip install sentence-transformers
pip install xgboost
pip install streamlit
```

---

# **2. Start the Streamlit App**

Run:

```bash
streamlit run App3.py
```

You should see output like:

```
Local URL: http://localhost:8501
Network URL: http://<your-ip>:8501
```

Open the URL in your browser.

---

# 🧠 How to Use the App

### **Step 1 — Enter Demographic Information**

Fill in:

* Age
* Gender
* Race
* Ethnicity

These features help the model include structured inputs alongside semantic embeddings.

---

### **Step 2 — Select a Job Role & Input Job Description**

Choose from the dropdown list and copy the job description texts.
The app automatically loads the **standard job description** and **job description texts** used during training.

---

### **Step 3 — Upload Resume**



Upload your `.pdf` file — the app will automatically extract text.

---

### **Step 4 — Predict Match Probability**

Click:

```
🔮 Predict Match Probability
```

The app will:

1. Generate BERT embeddings
2. Compute semantic similarity
3. Build the full feature vector
4. Predict with the trained XGBoost model
5. Display:

   * **Match / Not Match**
   * **Probability score**
   * **Confidence bar chart**
   * **If not match: AI Gerated suggestions**

---

# 📊 Fairness & Data Insights

The second tab includes interactive visualizations such as:

* Match probability distribution by **Race**
* Match probability distribution by **Gender**
* Match probability distribution by **Age group**
* Detailed subgroup views

These tools help evaluate model fairness and detect bias across demographic attributes.

---

# 📁 Project Structure

```
project/
│── App3.py                  # Streamlit application
│── xgb_model.pkl            # Trained XGBoost model
│── preprocessor.pkl         # Saved OneHotEncoder + Scaler
│── job_applicant_dataset.csv
│── requirements.txt
│── README.md
```

---

# 📝 Notes

* Sentence-BERT runs on **CPU**, no GPU required.
* PDF extraction uses `pdfminer.six` or `PyPDF2` depending on your setup.
* Hard-negative training significantly improved generalization—especially for rejecting irrelevant resumes.
