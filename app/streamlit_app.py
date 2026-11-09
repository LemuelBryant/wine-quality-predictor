import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #777777;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .result-card {
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.35);
            text-align: center;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .result-label {
            font-size: 1rem;
            color: #888888;
        }

        .result-score {
            font-size: 3rem;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            padding: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load trained models
# --------------------------------------------------


@st.cache_resource
def load_models():
    red_model = joblib.load("models/best_random_forest_red_wine.pkl")

    white_model = joblib.load("models/best_random_forest_white_wine.pkl")

    return red_model, white_model


red_model, white_model = load_models()


# --------------------------------------------------
# Application header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🍷 Wine Quality Predictor</div>', unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Estimate wine quality using machine learning and
        laboratory measurements of wine composition.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("About This Project")

    st.write("""
        This application uses trained Random Forest
        regression models to estimate wine quality.

        Separate models were developed for red and
        white wines using physicochemical measurements.
        """)

    st.divider()

    st.subheader("How It Works")

    st.write("""
        1. Select a wine type.
        2. Enter its laboratory measurements.
        3. Submit the measurements.
        4. Review the predicted quality score.
        """)

    st.divider()

    st.caption("Developed as a machine learning portfolio project.")


# --------------------------------------------------
# Wine selection
# --------------------------------------------------

wine_type = st.radio("Select Wine Type", ["Red Wine", "White Wine"], horizontal=True)

st.write("Enter the wine's physicochemical measurements below.")


# --------------------------------------------------
# Measurement input form
# --------------------------------------------------

with st.form("wine_prediction_form"):

    st.subheader("Wine Measurements")

    col1, col2, col3 = st.columns(3)

    # Acidity and sugar
    with col1:

        st.markdown("**Acidity & Sugar**")

        fixed_acidity = st.number_input(
            "Fixed Acidity",
            min_value=0.0,
            value=7.0,
            step=0.1,
            help="Nonvolatile acids in the wine (g/dm³).",
        )

        volatile_acidity = st.number_input(
            "Volatile Acidity",
            min_value=0.0,
            value=0.5,
            step=0.01,
            help="Primarily acetic acid (g/dm³).",
        )

        citric_acid = st.number_input(
            "Citric Acid",
            min_value=0.0,
            value=0.3,
            step=0.01,
            help="Citric acid concentration (g/dm³).",
        )

        residual_sugar = st.number_input(
            "Residual Sugar",
            min_value=0.0,
            value=2.0,
            step=0.1,
            help="Sugar remaining after fermentation (g/dm³).",
        )

    # Sulfur dioxide and chlorides
    with col2:

        st.markdown("**Sulfur Dioxide & Chlorides**")

        chlorides = st.number_input(
            "Chlorides",
            min_value=0.0,
            value=0.08,
            step=0.001,
            format="%.3f",
            help="Sodium chloride concentration (g/dm³).",
        )

        free_sulfur_dioxide = st.number_input(
            "Free Sulfur Dioxide",
            min_value=0.0,
            value=15.0,
            step=1.0,
            help="Free sulfur dioxide concentration (mg/dm³).",
        )

        total_sulfur_dioxide = st.number_input(
            "Total Sulfur Dioxide",
            min_value=0.0,
            value=46.0,
            step=1.0,
            help="Total sulfur dioxide concentration (mg/dm³).",
        )

        sulphates = st.number_input(
            "Sulphates",
            min_value=0.0,
            value=0.65,
            step=0.01,
            help="Sulphate concentration (g/dm³).",
        )

    # Other measurements
    with col3:

        st.markdown("**Other Measurements**")

        density = st.number_input(
            "Density",
            min_value=0.0,
            value=0.996,
            step=0.001,
            format="%.4f",
            help="Wine density (g/cm³).",
        )

        pH = st.number_input(
            "pH",
            min_value=0.0,
            max_value=14.0,
            value=3.3,
            step=0.1,
            help="Acidity or alkalinity measurement.",
        )

        alcohol = st.number_input(
            "Alcohol",
            min_value=0.0,
            value=10.0,
            step=0.1,
            help="Alcohol content (% by volume).",
        )

    st.divider()

    submitted = st.form_submit_button(
        "Predict Wine Quality", type="primary", width="stretch"
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

    if total_sulfur_dioxide == 0:

        st.error("Total sulfur dioxide must be greater than zero.")

    elif free_sulfur_dioxide > total_sulfur_dioxide:

        st.error("Free sulfur dioxide cannot exceed total sulfur dioxide.")

    else:

        # Calculate engineered feature
        so2_ratio = free_sulfur_dioxide / total_sulfur_dioxide

        # Prepare model input
        input_data = pd.DataFrame(
            [
                {
                    "fixed acidity": fixed_acidity,
                    "volatile acidity": volatile_acidity,
                    "citric acid": citric_acid,
                    "residual sugar": residual_sugar,
                    "chlorides": chlorides,
                    "free sulfur dioxide": free_sulfur_dioxide,
                    "total sulfur dioxide": total_sulfur_dioxide,
                    "density": density,
                    "pH": pH,
                    "sulphates": sulphates,
                    "alcohol": alcohol,
                    "so2_ratio": so2_ratio,
                }
            ]
        )

        # Select appropriate model
        if wine_type == "Red Wine":
            model = red_model
        else:
            model = white_model

        # Generate prediction
        prediction = model.predict(input_data)[0]

        # Display results
        st.divider()

        st.subheader("Prediction Results")

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    Estimated {wine_type} Quality
                </div>
                <div class="result-score">
                    {prediction:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "The predicted value is a regression estimate "
            "on the original dataset's quality-score scale."
        )

        # Show calculated feature
        with st.expander("View Calculated Feature"):

            st.write(
                "The model uses the ratio of free sulfur "
                "dioxide to total sulfur dioxide."
            )

            st.metric("SO₂ Ratio", f"{so2_ratio:.4f}")

            st.dataframe(input_data, width="stretch", hide_index=True)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption("""
    **Disclaimer:** This model estimates wine quality based
    on patterns learned from laboratory measurements.
    Its predictions are not a substitute for professional
    sensory evaluation or wine tasting.
    """)
