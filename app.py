import streamlit as st
import joblib
import pandas as pd
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and vectorizer
final_classifier = joblib.load(os.path.join(BASE_DIR, "model/gender_model_v2.joblib"))
final_vectorizer = joblib.load(os.path.join(BASE_DIR, "model/vectorizer.joblib"))

FEEDBACK_PATH = os.path.join(BASE_DIR, "feedback-identification.csv")
VALID_LABELS = {"male", "female", "common"}

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


# Feedback override
def get_feedback_override(name):
    if not os.path.exists(FEEDBACK_PATH):
        return None

    df = pd.read_csv(FEEDBACK_PATH)
    df = df[df["Human_Verdict"].isin(VALID_LABELS)]

    match = df[df["Name"].str.lower() == name.lower()]
    if not match.empty:
        return match.iloc[-1]["Human_Verdict"]

    return None


# Prediction
def predict_gender(name, threshold=0.80, margin=0.65):

    feedback_verdict = get_feedback_override(name)
    if feedback_verdict is not None:
        return feedback_verdict, 1.0

    features = gender_features(name)
    vec = final_vectorizer.transform([features])

    probs = final_classifier.predict_proba(vec)[0]
    classes = [str(c) for c in final_classifier.classes_]

    prob_dict = dict(zip(classes, probs))
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    top_label, top_prob = sorted_probs[0]
    second_prob = sorted_probs[1][1]

    if top_prob < threshold or (top_prob - second_prob) < margin:
        return "common", float(top_prob)

    return top_label, float(top_prob)


# Store feedback (same-row update)
def store_feedback(name, model_prediction, human_verdict, confidence, thumb=None):

    columns = [
        "Name", "Model_Prediction", "Human_Verdict", "Confidence",
        "male", "female", "common"
    ]

    if os.path.exists(FEEDBACK_PATH):
        df = pd.read_csv(FEEDBACK_PATH)
    else:
        df = pd.DataFrame(columns=columns)

    match_idx = df[df["Name"].str.lower() == name.lower()].index

    if len(match_idx) > 0:
        idx = match_idx[-1]

        if thumb == "up":
            df.at[idx, model_prediction] += 1  # increment only

        if human_verdict in VALID_LABELS:
            df.at[idx, "Human_Verdict"] = human_verdict

        df.at[idx, "Model_Prediction"] = model_prediction
        df.at[idx, "Confidence"] = confidence

    else:
        male = female = common = 0

        if thumb == "up":
            locals()[model_prediction] = 1  # increment only

        df = pd.concat([
            df,
            pd.DataFrame([[
                name,
                model_prediction,
                human_verdict if human_verdict in VALID_LABELS else "",
                confidence,
                male, female, common
            ]], columns=columns)
        ], ignore_index=True)

    df.to_csv(FEEDBACK_PATH, index=False)


# ---------------- STREAMLIT APP ----------------

st.title("Gender Identification – Human Feedback System")

name = st.text_input("Enter a name")

# Session state
if "result" not in st.session_state:
    st.session_state.result = None
if "show_expert" not in st.session_state:
    st.session_state.show_expert = False

# Predict
if st.button("Predict"):
    if name.strip():
        pred, conf = predict_gender(name)
        st.session_state.result = {
            "name": name,
            "prediction": pred,
            "confidence": conf
        }
        st.session_state.show_expert = False

# Display Prediction
if st.session_state.result:
    r = st.session_state.result

    st.subheader("Model Prediction")
    st.write(f"**Name:** {r['name']}")
    st.write(f"**Prediction:** {r['prediction']}")
    st.write(f"**Confidence:** {r['confidence']:.2f}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Thumbs Up"):
            store_feedback(
                r["name"], r["prediction"], r["prediction"],
                r["confidence"], thumb="up"
            )
            st.success("Thumbs up recorded")

    with col2:
        if st.button("👎 Thumbs Down"):
            store_feedback(
                r["name"], r["prediction"], "",
                r["confidence"]
            )
            st.session_state.show_expert = True
            st.warning("Please provide correct classification")

    # Show expert verdict ONLY after thumbs down
    if st.session_state.show_expert:
        st.subheader("Human Expert Verdict")
        verdict = st.radio(
            "Select correct classification:",
            ["male", "female", "common"],
            horizontal=True,
            key="expert_verdict"
        )

        if st.button("Save Expert Feedback"):
            store_feedback(
                r["name"], r["prediction"], verdict,
                r["confidence"]
            )
            st.success("Expert feedback saved")
            st.session_state.show_expert = False

# View feedback
if st.checkbox("Show stored feedback"):
    if os.path.exists(FEEDBACK_PATH):
        st.dataframe(pd.read_csv(FEEDBACK_PATH))
    else:
        st.info("No feedback collected yet.")
