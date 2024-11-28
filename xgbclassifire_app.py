import streamlit as st
import pickle
import numpy as np

# Load the trained XGBoostClassifier model
model_path = "xgb_model.pkl"  
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Define the renamed features used in the model
RENAMED_FEATURES = ['F1_tetha/gamma', 'F2_tetha/alpha', 'F2_tetha/beta', 'F2_tetha/gamma', 'T2_tetha/gamma', 
                    'P1_tetha/gamma', 'P2_tetha/beta', 'P2_tetha/gamma', 'O2_tetha/gamma', 'Epileptic_event_frequency']

# Map the renamed features to their original feature names
ORIGINAL_FEATURES = ['F1_th/gama', 'F2_th/al', 'F2_th/beta', 'F2_th/gama','T2_th/gama', 
                     'P1_th/gama', 'P2_th/beta','P2_th/gama','O2_th/gama', 'Num_per_h']                   
                     

feature_mapping = dict(zip(RENAMED_FEATURES, ORIGINAL_FEATURES))

# Streamlit App Title
st.title("Epileptic Event Prediction")
st.write("Enter values for the features below to predict treatment efficacy.")

# Input Section
st.subheader("Input Features")

# Initialize dictionary to store user input
user_input = {}

# Create layout with rows of 3 columns
columns = st.columns(3)

for idx, renamed_feature in enumerate(RENAMED_FEATURES):
    col = columns[idx % 3]
    user_input[renamed_feature] = col.number_input(f"{renamed_feature}:", value=0.0)

# Convert user input to a format suitable for the model
input_array = np.array([list(user_input.values())])

# Style for the button
button_style = """
    <style>
    .stButton > button {
        font-weight: bold;
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 8px 16px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        cursor: pointer;
        border-radius: 4px;
    }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Prediction Section
if st.button("Predict"):
    if 'xgb_model' in locals():
        prediction = xgb_model.predict(input_array)
        prediction_proba = xgb_model.predict_proba(input_array)

        # Display results
        st.subheader("Prediction Result")
        if prediction[0] == 0:
            st.success("Prediction: No Sulforaphane Treatment (Pilocarpine Alone)")
        else:
            st.success("Prediction: Sulforaphane Treatment")

        st.subheader("Prediction Probabilities")
        st.write(f"Probability of No Treatment: {prediction_proba[0][0]:.2f}")
        st.write(f"Probability of Sulforaphane Treatment: {prediction_proba[0][1]:.2f}")
    else:
        st.error("Model is not loaded. Please check the model file path.")