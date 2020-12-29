import numpy as np
import pickle
import pandas as pd
import streamlit as st
import time

pickle_in = open("model.pickle", "rb")
classifier = pickle.load(pickle_in)


def get_loan_prediction(Loan_ID,Gender,Married,Dependents,Education,Self_Employed,LoanAmount,Loan_Amount_Term,ApplicantIncome,CoapplicantIncome,Credit_History,Property_Area):
    if Property_Area =='Semiurban':
        Rural,Semiurban,Urban, = 0,1,0
    elif Property_Area =='Rural':
        Rural, Semiurban, Urban, = 1, 0, 0
    else:
        Rural, Semiurban, Urban, = 0, 0, 1


    X = pd.DataFrame({'Loan_ID': Loan_ID,'Gender': Gender, 'Married': Married,
                      'Dependents':Dependents,'Education':Education,'Self_Employed': Self_Employed,
                      'ApplicantIncome': ApplicantIncome,'CoapplicantIncome': CoapplicantIncome,
                      'LoanAmount':LoanAmount,'Loan_Amount_Term': Loan_Amount_Term,
                      'Credit_History': Credit_History,'Property_Area_Rural':Rural,'Property_Area_Semiurban':Semiurban,'Property_Area_Urban':Urban},index =[1])

    def Transforming(df):
        df['Dependents'] = df['Dependents'].str.rstrip('+')
        df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0}).astype(np.int)
        df['Married'] = df['Married'].map({'No': 0, 'Yes': 1}).astype(np.int)
        df['Education'] = df['Education'].map({'Not Graduate': 0, 'Graduate': 1}).astype(np.int)
        df['Self_Employed'] = df['Self_Employed'].map({'No': 0, 'Yes': 1}).astype(np.int)
        df['Credit_History'] = df['Credit_History'].map({'No':0,'Yes':1}).astype(np.int)
        df['Dependents'] = df['Dependents'].astype(np.int)
        return df
    X = Transforming(X)
    prediction = classifier.predict(X)[0]
    return prediction


def main():
    st.markdown(
        """
        <style>
            .reportview-container {background-color:Pink;}
        </style>
        """,
        unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color:Yellow;padding:10px">
        <h2 style="color:Black;text-align:center;">Loan Approval Prediction App </h2>
    </div>
    """, unsafe_allow_html=True)

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    Loan_ID = st.number_input("Enter Your Loan Application ID",1000,100000,step = 1)
    Gender = st.radio("Gender",('Male','Female'))
    Married = st.radio("Married",('No','Yes'))
    Dependents = st.radio("Number of Children",('0','1','2','3+'))
    Education = st.radio("Education Level",('Graduate','Not Graduate'))
    Self_Employed = st.radio("Self Employed",('Yes','No'))
    ApplicantIncome = st.number_input("Current Applicants Income",0.,100000.,step = 1.)
    CoapplicantIncome = st.number_input("Co-Applicants Income(if any)",0.0,100000.0,step = 1.0)
    LoanAmount = st.number_input("Loan Amount", 0.0,100000.0,step = 1.00)
    Loan_Amount_Term = st.select_slider("Loan Period (in number of Days)",options = [12,36,60,84,120,180,240,300,360,480])
    Credit_History = st.radio("Credit History",('Yes','No'))
    Property_Area = st.radio("Property Area",("Rural",'Semiurban','Urban'))
    result = ""
    if st.button("Predict"):
        result = get_loan_prediction(Loan_ID,Gender,Married,Dependents,Education,Self_Employed,LoanAmount,Loan_Amount_Term,ApplicantIncome,CoapplicantIncome,Credit_History,Property_Area)

    if result == 1:
        with st.spinner('Wait for it...'):
            time.sleep(2)
            st.success('Loan Approved')
            st.balloons()
    elif result == 0:
        with st.spinner('Wait for it..'):
            time.sleep(1)
        st.success('Loan Rejected')
    else:
        st.success('Decision Pending')

    st.markdown("""<h4 style="color:Black;text-align:center;">Hello from Pinky Chaudhary</h2>""", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
