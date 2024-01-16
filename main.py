import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import pickle

# Load the trained model
pickle_in = open('stroke/stroke.pkl', 'rb')
model = pickle.load(pickle_in)

def predict_stroke():
    st.sidebar.header('Stroke Prediction')
    st.title('Stroke Prediction')
    st.markdown('According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.\nThus the objective is to predicts the likelihood of having a stroke based on certain health parameters.')
    st.markdown('Several constraints were placed on the selection of these instances from a larger database.')

    name = st.text_input("Name:")

    gender = st.radio("Gender", ["Female", "Male"])

    age = st.number_input("Age:", min_value=0, max_value=120, value=30)
    st.markdown('Age: Age (years)')

    hypertension = st.checkbox("Hypertension")
    heart_disease = st.checkbox("Heart Disease")

    ever_married = st.radio("Ever Married", ["No", "Yes"])

    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job"])

    Residence_type = st.radio("Residence Type", ["Urban", "Rural"])

    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=80.0)

    bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):")
    st.markdown('BMI: Body mass index (weight in kg/(height in m)^2)')

    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

    # Check if all fields are filled
    if not (name and gender and age and ever_married and work_type and Residence_type and avg_glucose_level and bmi and smoking_status):
        st.warning("Please fill in all the fields.")
        return

    # Convert input to numeric values
    gender = 1 if gender == "Male" else 0
    ever_married = 1 if ever_married == "Yes" else 0

    work_type_mapping = {"Private": 0, "Self-employed": 1, "Govt_job": 2}
    work_type = work_type_mapping.get(work_type, 0)

    Residence_type = 0 if Residence_type == "Urban" else 1

    smoking_status_mapping = {"never smoked": 1, "formerly smoked": 0, "smokes": 2}
    smoking_status = smoking_status_mapping.get(smoking_status, 0)

    submit = st.button('Predict')

    if submit:
        input_data = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])
        prediction = model.predict(input_data)
        if prediction == 0:
            st.write('Low risk of stroke. Stay healthy!')
        else:
            st.write(name, "High risk of stroke. Please consult with a healthcare professional.")
            st.markdown('[Visit Here](https://www.who.int/home/search-results?indexCatalogue=genericsearchindex1&searchQuery=stroke&wordsMode=AnyWord) to know more about stroke.)')


def main():
    new_title = '<p style="font-size: 42px;">Welcome The Stroke Prediction App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)
    read_me = st.markdown("""
    The application is built using Streamlit  
    to demonstrate Stroke Prediction. It performs prediction on multiple parameters
    [here](https://github.com/sOR-o/stroke-predictor).""")
    st.sidebar.title("Select Activity")
    choice = st.sidebar.selectbox(
        "MODE", ("About", "Predict Stroke"))
    if choice == "Predict Stroke":
        read_me_0.empty()
        read_me.empty()
        predict_stroke()
    elif choice == "About":
        print()


if __name__ == '__main__':
    main()
