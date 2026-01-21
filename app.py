import streamlit as st
import joblib
import pandas as pd
import os
from datetime import datetime

# Base directory (works in Streamlit + Python)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and vectorizer
final_classifier = joblib.load(os.path.join(BASE_DIR, "model/gender_model.joblib"))
final_vectorizer = joblib.load(os.path.join(BASE_DIR, "model/vectorizer.joblib"))

# Feature extraction
def gender_features(name):
    name = name.lower()
    features = {}

    features["first_letter"] = name[0]
    features["last_letter"] = name[-1]
    features["name_length"] = len(name)
    features["ends_with_vowel"] = name[-1] in "aeiou"
    features["starts_with_vowel"] = name[0] in "aeiou"

    padded = f"<{name}>"
    for n in (2, 3, 4):
        for i in range(len(padded) - n + 1):
            gram = padded[i:i+n]
            features[f"char_{n}gram_{gram}"] = True

    return features

# Prediction with "Common" logic
def predict_gender(name, threshold=0.85, margin=0.65):
    features = gender_features(name)
    vec = final_vectorizer.transform([features])

    probs = final_classifier.predict_proba(vec)[0]
    classes = [str(c) for c in final_classifier.classes_]

    prob_dict = dict(zip(classes, probs))
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    top_label, top_prob = sorted_probs[0]
    second_prob = sorted_probs[1][1]

    if top_prob < threshold or (top_prob - second_prob) < margin:
        return "Common", float(top_prob), prob_dict

    return top_label, float(top_prob), prob_dict

# Store feedback
def store_feedback(name, model_prediction, human_verdict, confidence):
    file_path = os.path.join(BASE_DIR, "feedback-identification.csv")

    row = pd.DataFrame(
        [[name, model_prediction, human_verdict, confidence]],
        columns=[
            "Name",
            "Model_Prediction",
            "Human_Verdict",
            "Confidence"
        ]
    )

    if os.path.exists(file_path):
        row.to_csv(file_path, mode="a", header=False, index=False)
    else:
        row.to_csv(file_path, index=False)

# Streamlit UI
st.title("Gender Identification – Human Feedback System")

name = st.text_input("Enter a name")

# ---------------- Predict ----------------
if st.button("Predict"):
    if name.strip() == "":
        st.warning("Please enter a valid name.")
    else:
        prediction, confidence, probs = predict_gender(name)

        # Save result in session state
        st.session_state["result"] = {
            "name": name,
            "prediction": prediction,
            "confidence": confidence,
            "probs": probs
        }

#Display Prediction 
if "result" in st.session_state:
    result = st.session_state["result"]

    st.subheader("Model Prediction")
    st.write(f"**Name:** {result['name']}")
    st.write(f"**Prediction:** {result['prediction']}")
    st.write(f"**Confidence:** {result['confidence']:.2f}")
    st.json({k: round(v, 3) for k, v in result["probs"].items()})

    st.subheader(" Human Expert Verdict")

    verdict = st.radio(
        "Select correct classification:",
        ["male", "female", "common"],
        key="verdict_radio",
        horizontal=True
    )

    # Save Feedback 
    if st.button("Save Feedback"):
        store_feedback(
            name=result["name"],
            model_prediction=result["prediction"],
            human_verdict=verdict,
            confidence=result["confidence"]
        )

        st.success("Feedback saved successfully!")

        # Clear session state so next prediction is clean
        del st.session_state["result"]

# Optional: View feedback
if st.checkbox("Show stored feedback"):
    feedback_file = os.path.join(BASE_DIR, "feedback-identification.csv")
    if os.path.exists(feedback_file):
        st.dataframe(pd.read_csv(feedback_file))
    else:
        st.info("No feedback collected yet.")
