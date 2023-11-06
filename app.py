import streamlit as st
import pandas as pd
import os

#Import profiling capability
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

#ML
from pycaret.regression import setup, compare_models, pull, save_model




with st.sidebar:
    st.title("AutoML")
    choice =  st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This Applicaton allows you to build an automated ML pipeline using Streamlit, Pandas Profiling and Pycaret.")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)


if choice == "Upload":
    st.title("Upload your dataset for modeling!")
    file = st.file_uploader("Upload your dataset here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
    
if choice == "Profiling":
    st.title("Automated Exploratory data analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "ML":
    st.title("Machine Learning go -->>")
    target = st.selectbox("Select your Target", df.columns)
    if st.button("Train model"):
        setup_df = setup(df, target=target)

        st.info("this is the MLExperiment settings")
        st.write(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')

if choice == "Download":
   with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
