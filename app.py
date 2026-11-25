
import os
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Titanic AI | Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

FEATURES = ["pclass", "sex", "age", "fare"]

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #101827;
        color: #F8FAFC;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: 1px;
    }

    .subtitle {
        color: #AAB7CA;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .result-card {
        background: #202E42;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #35445A;
        text-align: center;
    }

    .result-value {
        font-size: 42px;
        font-weight: bold;
        color: #70D7AA;
    }

    .result-label {
        color: #AAB7CA;
        font-size: 13px;
    }

    div.stButton > button {
        width: 100%;
        background-color: #269B79;
        color: white;
        border-radius: 10px;
        height: 48px;
        font-weight: bold;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #31B58F;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists("model.joblib"):
        raise FileNotFoundError(
            "The model file model.joblib was not found."
        )

    return joblib.load("model.joblib")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">TITANIC AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'SURVIVAL PREDICTION SYSTEM | MACHINE LEARNING DEMO'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader("Passenger Information")

st.write(
    "Enter passenger details to generate a survival prediction."
)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "First Class",
            2: "Second Class",
            3: "Third Class"
        }[x]
    )

    sex = st.radio(
        "Gender",
        options=["female", "male"],
        horizontal=True
    )

with col2:

    age = st.slider(
        "Passenger Age",
        min_value=1,
        max_value=80,
        value=30
    )

    fare = st.slider(
        "Ticket Fare",
        min_value=0.0,
        max_value=500.0,
        value=30.0,
        step=1.0
    )

st.divider()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("PREDICT SURVIVAL"):

    try:
        model = load_model()

        # Encode gender exactly as in Project 04
        sex_encoded = 1 if sex == "female" else 0

        # Maintain the original training feature order
        input_data = pd.DataFrame(
            [[pclass, sex_encoded, age, fare]],
            columns=FEATURES
        )

        # Validate inputs
        if not input_data.columns.tolist() == FEATURES:
            raise ValueError("Invalid feature order.")

        if not input_data.notna().all().all():
            raise ValueError("Please enter valid passenger details.")

        # Predict class and probability
        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        classes = list(model.classes_)

        survived_index = classes.index(1)

        survival_probability = probabilities[survived_index]

        # Display result
        st.subheader("Prediction Result")

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                "Estimated Survival Probability",
                f"{survival_probability:.1%}"
            )

        with metric_col2:
            st.metric(
                "Model Prediction",
                "Survived" if prediction == 1
                else "Did Not Survive"
            )

        st.progress(float(survival_probability))

        if prediction == 1:
            st.success(
                "The model predicts that this passenger survived."
            )
        else:
            st.warning(
                "The model predicts that this passenger did not survive."
            )

    except FileNotFoundError:
        st.error(
            "The model file is missing. "
            "Please train and save the model first."
        )

    except (ValueError, IndexError) as error:
        st.error(f"Unable to process prediction: {error}")

    except Exception:
        st.error(
            "Something went wrong while generating the prediction. "
            "Please check the application files and model."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Educational Machine Learning project using a Decision Tree "
    "trained on historical Titanic passenger data. "
    "Predictions are estimates from a historical dataset, "
    "not factual conclusions or advice."
)

st.caption(
    "InternCircle | Project 05 | Data Science & AI"
)
