
import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
from pathlib import Path
import math

# =====================================
# 1. PAGE CONFIGURATION
# =====================================
st.set_page_config(
    page_title="AI Based Placement Predictor & Advisor",
    page_icon="🎓",
    layout="wide"
)

# =====================================
# 2. LOAD MODEL AND ENCODER
# =====================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "placement_model.pkl"
ENCODER_PATH = BASE_DIR / "ordinal_encoder.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    explainer = shap.TreeExplainer(model)
    return model, encoder, explainer


if not MODEL_PATH.is_file() or not ENCODER_PATH.is_file():
    st.error("Model or encoder file not found.")
    st.write("App folder:", str(BASE_DIR))
    st.write("Model exists:", MODEL_PATH.is_file())
    st.write("Encoder exists:", ENCODER_PATH.is_file())
    st.stop()

try:
    model, encoder, explainer = load_artifacts()
except Exception as e:
    st.error(f"Could not load model or encoder: {e}")
    st.stop()


# =====================================
# 3. VALIDATE MODEL FEATURES
# =====================================
categorical_column = "Internship_Experience"

feature_names = list(model.feature_names_in_)

expected_features = [
    "Communication_Skills",
    "CGPA",
    "Prev_Sem_Result",
    "IQ",
    "Projects_Completed",
    "Extra_Curricular_Score",
    "Internship_Experience",
    "Academic_Performance"
]

if set(feature_names) != set(expected_features):
    st.error("Model features do not match the expected features.")
    st.write("Model features:", feature_names)
    st.stop()

if list(encoder.feature_names_in_) != [categorical_column]:
    st.error("Encoder must be trained on Internship_Experience only.")
    st.write("Encoder features:", list(encoder.feature_names_in_))
    st.stop()

internship_options = list(encoder.categories_[0])

feature_labels = {
    "Communication_Skills": "Communication Skills",
    "CGPA": "CGPA",
    "Prev_Sem_Result": "Previous Semester Result",
    "IQ": "Aptitude Score",
    "Projects_Completed": "Projects Completed",
    "Extra_Curricular_Score": "Extra-curricular Score",
    "Internship_Experience": "Internship Experience",
    "Academic_Performance": "Academic Performance"
}


# =====================================
# 4. APPLICATION HEADER
# =====================================
st.title("🎓 AI Based Placement Predictor & Advisor")

st.write(
    "Enter your details to estimate your placement probability "
    "and discover your strengths and areas for improvement."
)

st.caption(
    "This prediction is based on your trained machine-learning "
    "model and does not guarantee actual placement."
)

st.divider()


# =====================================
# 5. STUDENT INPUT FORM
# =====================================
st.subheader("Student Profile")

with st.form("student_form"):

    col1, col2 = st.columns(2)

    with col1:
        communication = st.number_input(
            "Communication Skills (0–10)",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.5
        )

        cgpa = st.number_input(
            "CGPA (0–10)",
            min_value=0.0,
            max_value=10.0,
            value=8.0,
            step=0.1
        )

        prev_sem = st.number_input(
            "Previous Semester Result (0–10)",
            min_value=0.0,
            max_value=10.0,
            value=8.0,
            step=0.1
        )

        # User can type an aptitude score from 0 to 200
        aptitude = st.number_input(
            "Aptitude Score (0–200)",
            min_value=0,
            max_value=200,
            value=100,
            step=1
        )

    with col2:
        projects = st.number_input(
            "Projects Completed",
            min_value=0,
            max_value=100,
            value=2,
            step=1
        )

        extracurricular = st.number_input(
            "Extra-curricular Score (0–10)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5
        )

        internship = st.selectbox(
            "Internship Experience",
            options=internship_options
        )

        academic = st.number_input(
            "Academic Performance (0–10)",
            min_value=0.0,
            max_value=10.0,
            value=8.0,
            step=0.5
        )

    submitted = st.form_submit_button(
        "Predict Placement",
        type="primary",
        use_container_width=True
    )


# =====================================
# 6. PREDICTION
# =====================================
if submitted:

    student_data = {
        "Communication_Skills": communication,
        "CGPA": cgpa,
        "Prev_Sem_Result": prev_sem,
        "IQ": aptitude,
        "Projects_Completed": projects,
        "Extra_Curricular_Score": extracurricular,
        "Internship_Experience": internship,
        "Academic_Performance": academic
    }

    student = pd.DataFrame([student_data])

    try:
        # Encode only Internship_Experience
        student[categorical_column] = encoder.transform(
            student[[categorical_column]]
        )[:, 0]

        # Maintain the model's training feature order
        student = student[feature_names]

        # Predict class and probabilities
        prediction = model.predict(student)[0]
        probabilities = model.predict_proba(student)[0]
        classes = list(model.classes_)

        # Assumes 1 = Placed and 0 = Not Placed
        if 1 not in classes:
            st.error(f"Class 1 was not found. Model classes: {classes}")
            st.stop()

        positive_index = classes.index(1)
        placement_probability = float(probabilities[positive_index])
        is_placed = prediction == 1

        # =====================================
        # 7. PLACEMENT RESULT
        # =====================================
        st.divider()
        st.subheader("Placement Prediction")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if is_placed:
                st.success("Prediction: Placed")
            else:
                st.warning("Prediction: Not Placed")

        with result_col2:
            st.metric(
                "Estimated Placement Probability",
                f"{placement_probability:.1%}"
            )

        st.progress(placement_probability)

        st.caption(
            "This probability comes from your trained model. "
            "It may not represent a calibrated real-world probability."
        )

        # =====================================
        # 8. SHAP EXPLANATIONS
        # =====================================
        st.divider()
        st.subheader("Your Strengths & Weaknesses")

        st.write(
            "These sections show how each feature influenced the "
            "model's prediction for the placed class. They describe "
            "model contributions, not proof of a student's actual ability."
        )

        try:
            shap_explanation = explainer(student)
            shap_values = np.asarray(shap_explanation.values)

            if shap_values.ndim == 3:
                contributions = shap_values[0, :, positive_index]
            elif shap_values.ndim == 2:
                contributions = shap_values[0]
            else:
                raise ValueError(
                    f"Unexpected SHAP output shape: {shap_values.shape}"
                )

            explanation_df = pd.DataFrame({
                "Feature": [
                    feature_labels.get(name, name)
                    for name in feature_names
                ],
                "Contribution": contributions
            })

            # Positive contributions, strongest first
            strengths = explanation_df[
                explanation_df["Contribution"] > 0
            ].sort_values("Contribution", ascending=False).reset_index(drop=True)

            # Negative contributions, strongest negative first
            weaknesses = explanation_df[
                explanation_df["Contribution"] < 0
            ].sort_values("Contribution", ascending=True).reset_index(drop=True)

            # Divide positive contributions into relative groups.
            # This avoids fixed SHAP thresholds that may not suit
            # every model or dataset.
            strength_groups = {
                "Very Good": [],
                "Good": [],
                "Average": []
            }

            n = len(strengths)

            for i, row in strengths.iterrows():
                if n == 0:
                    break

                if i < math.ceil(n / 3):
                    group = "Very Good"
                elif i < math.ceil(2 * n / 3):
                    group = "Good"
                else:
                    group = "Average"

                strength_groups[group].append(row["Feature"])

            left, right = st.columns(2)

            # =====================================
            # STRENGTHS
            # =====================================
            with left:
                st.markdown("### 💪 Strengths")

                if strengths.empty:
                    st.info(
                        "No features positively contributed to "
                        "the placed-class prediction."
                    )
                else:
                    for group, features in strength_groups.items():
                        if features:
                            st.markdown(f"**{group}**")

                            for feature in features:
                                st.write(f"✓ {feature}")

            # =====================================
            # WEAKNESSES
            # =====================================
            with right:
                st.markdown("### 🎯 Need Improvement")

                if weaknesses.empty:
                    st.success(
                        "No features negatively contributed to "
                        "the placed-class prediction."
                    )
                else:
                    for _, row in weaknesses.iterrows():
                        st.write(f"• {row['Feature']}")

            st.caption(
                "Very Good, Good and Average rank the positive "
                "contributions relative to this student's other "
                "positive contributions. They are not absolute skill "
                "ratings. A negative contribution does not automatically "
                "mean the student's actual skill is poor."
            )

        except Exception as shap_error:
            st.warning(
                "Placement prediction succeeded, but the explanation "
                f"could not be displayed: {shap_error}"
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check the model features, encoder categories, "
            "and saved model version."
        )