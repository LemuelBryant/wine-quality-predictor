import streamlit as st
import pandas as pd
import joblib

# Load trained models
red_model = joblib.load("models/best_random_forest_red_wine.pkl")
white_model = joblib.load("models/best_random_forest_white_wine.pkl")

# Page configuration
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

# Application title
st.title("🍷 Wine Quality Predictor")

st.write("""
    This application uses machine learning to estimate wine quality
    based on its physicochemical properties.
    
    Select a wine type and enter its laboratory measurements
    to receive a predicted quality score.
    """)

# Wine selection
wine_type = st.selectbox("Select Wine Type", ["Red Wine", "White Wine"])

st.subheader("Wine Characteristics")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, value=7.0)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, value=0.5)
    citric_acid = st.number_input("Citric Acid", min_value=0.0, value=0.3)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, value=2.0)
    chlorides = st.number_input("Chlorides", min_value=0.0, value=0.08)
    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide", min_value=0.0, value=15.0
    )

with col2:
    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide", min_value=0.0, value=46.0
    )
    density = st.number_input("Density", min_value=0.0, value=0.996)
    pH = st.number_input("pH", min_value=0.0, value=3.3)
    sulphates = st.number_input("Sulphates", min_value=0.0, value=0.65)
    alcohol = st.number_input("Alcohol", min_value=0.0, value=10.0)

if st.button("Predict Wine Quality"):

    # Calculate engineered feature
    if total_sulfur_dioxide == 0:
        st.error("Total sulfur dioxide must be greater than zero.")
    else:
        so2_ratio = free_sulfur_dioxide / total_sulfur_dioxide

        # Create input DataFrame
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

        # Select model
        if wine_type == "Red Wine":
            model = red_model
        else:
            model = white_model

        # Generate prediction
        prediction = model.predict(input_data)[0]

        st.success(f"Predicted Wine Quality: {prediction:.2f}")

        st.caption(
            "Wine quality is predicted on the dataset's " "original 0–10 quality scale."
        )
