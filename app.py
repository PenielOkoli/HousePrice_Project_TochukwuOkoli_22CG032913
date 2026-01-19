import streamlit as st
import pandas as pd
import pickle
import os

# 1. Page Configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# 2. Load the Model (with caching for performance)
@st.cache_resource
def load_model():
    # We use os.path to ensure it works on both Windows and Cloud (Linux)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', 'house_price_model.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

model = load_model()

# 3. App Header
st.title("🏠 House Price Prediction System")
st.markdown("### Project 3: Machine Learning Application")
st.markdown("Enter the house details below to estimate its sale price.")
st.divider()

if model is None:
    st.error("Error: Could not find 'house_price_model.pkl'. Please ensure it is in the 'model' folder.")
else:
    # 4. Input Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5, help="Rates the overall material and finish of the house")
            gr_liv_area = st.number_input("Living Area (sq ft)", min_value=300, max_value=10000, value=1500)
            garage_cars = st.selectbox("Garage Cars", [0, 1, 2, 3, 4])

        with col2:
            full_bath = st.number_input("Full Bathrooms", min_value=0, max_value=4, value=1)
            year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
            
            # Neighborhoods list (Must match the training data exactly)
            neighborhoods = [
                'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
                'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
                'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
                'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'
            ]
            neighborhood = st.selectbox("Neighborhood", sorted(neighborhoods))

        # Submit Button
        submitted = st.form_submit_button("Predict Price", type="primary")

    # 5. Prediction Logic
    if submitted:
        # Create a DataFrame with the inputs
        # IMPORTANT: The column names must match the training data exactly
        input_data = pd.DataFrame({
            'OverallQual': [overall_qual],
            'GrLivArea': [gr_liv_area],
            'GarageCars': [garage_cars],
            'FullBath': [full_bath],
            'YearBuilt': [year_built],
            'Neighborhood': [neighborhood]
        })

        # Make Prediction
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"Estimated Sale Price: **${prediction:,.2f}**")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.caption("Developed by [Your Name] | Matric No: [Your Matric No]")