
###
# Created on Fri Jun 27 13:46:31 2025
###
###############################################################################
# Machine Learning Model Deployment & Streamlit - Coding Our Web-App Part 1
###############################################################################

# IMPORT LIBRARIES

import streamlit as st
import pandas as pd
import joblib

# LOAD OUR MODEL PIPELIINE OBJECT
model = joblib.load("model.joblib")

# ADD TITLE AND INSTRUCTIONS
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood to purchase")


# OPEN ANACONDA PROMPT
# 1. Make sure that we are in our streamlit virtual environment 
## To do this, let us put: conda activate dsi-streamlit-web-app
## Then hit enter. We can see that we are now in that virtual environment: (dsi-streamlit-web-app) C:\Users\oleh_>

# 2. Then,type in: cd [our working directory path we copied] 
## (dsi-streamlit-web-app) C:\Users\oleh_> 
## type: cd C:\Users\oleh_\OneDrive\Desktop\Data Science Infinity\Machine Learning\Model Deployment\Streamlit
## Then hit enter. We are now pointing to the right place, where our web app script lives. 

# 3. Then, type in: streamlit run [the name of our web app script that we are building in python]
## (dsi-streamlit-web-app) C:\Users\oleh_\OneDrive\Desktop\Data Science Infinity\Machine Learning\Model Deployment\Streamlit>
## Type: streamlit run dsi-streamlit-web-app.py
 
## Code Note: If you get an error message such as click has no attribute get_os_args please do the following:
## Open up Anaconda Prompt, make sure you're in the correct environment, and upgrade the click package using this command:
## pip install click==8.0.4

## 4. Auto-open up a browser with our web app



###############################################################################
# Machine Learning Model Deployment & Streamlit - Coding Our Web-App Part 2
###############################################################################

# AGE INPUT FORM
age = st.number_input(
    label = "01. Enter the customer's age",
    min_value = 18,
    max_value = 120,
    value = 35
    )

# GENDER INPUT FORM
gender = st.radio(
    label = "02. Enter the customer's gender",
    options = ["M", "F"]
    )

# CREDIT SCORE INPUT FORM
credit_score = st.number_input(
    label = "03. Enter the customer's credit score",
    min_value = 0,
    max_value = 1000,
    value = 500
    )

# SUBMIT INPUTS TO MODEL
if st.button("Submit For Prediction"): 
    
    # STORE OUR DATA IN A DATAFRAME FOR PREDICTION
    new_data = pd.DataFrame({"age" : [age], "gender" : [gender], "credit_score" : [credit_score]})
    
    # APPLY MODEL PIPELINE TO THE INPUT DATA AND EXTRACT PROBABILITY PREDICTION
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # OUTPUT PREDICTION
    st.subheader(f"Based on these customer attributes, our model predicts a purchase probability of {pred_proba: .0%}")


