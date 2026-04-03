import streamlit as st
import pickle
import numpy as np

# ---------- Load Models ----------
model = pickle.load(open("logistic_regression_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ---------- Page Config ----------
st.set_page_config(page_title="Fake Job Detection", page_icon="💼", layout="centered")

st.title("💼 Fake Job Posting Detection System")
st.markdown("Enter job details below to check whether the job posting is **Real or Fake**.")

# ---------- Input Fields ----------
title = st.text_input("Job Title")
location = st.text_input("Location")
company = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Requirements")


st.markdown("---")

# ---------- Prediction ----------
if st.button("🔎 Predict Job Authenticity"):

    # combine all text fields (IMPORTANT)
    full_text = f"{title} {location} {company} {description} {requirements} {benefits}"

    if full_text.strip() == "":
        st.warning("⚠️ Please enter job information")
    else:
        # TFIDF transform
        vector = tfidf.transform([full_text])

        # prediction
        pred = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        confidence = np.max(prob) * 100

        label = label_encoder.inverse_transform([pred])[0]

        st.markdown("### 📊 Prediction Result")

        if label == 1 or label == "fake":
            st.error(f"⚠️ This job posting is **FAKE**  \nConfidence: {confidence:.2f}%")
        else:
            st.success(f"✅ This job posting is **REAL**  \nConfidence: {confidence:.2f}%")
