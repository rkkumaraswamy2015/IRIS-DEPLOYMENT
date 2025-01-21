import streamlit as st
import numpy as np
import pickle
import sklearn

st.markdown('<h1 style="color:green;">IRIS FLOWER PREDICTION APP</h1>', unsafe_allow_html=True)
with open("logreg_iris.pkl", "rb") as file:
       model=pickle.load(file)
st.write("Enter the feature details:")
sepal_length = st.slider("SEPAL_LENGTH", min_value=1.0, max_value=10.0, step=0.05)
sepal_width = st.slider("SEPAL_WIDTH", min_value=1.0, max_value=10.0, step=0.05)
petal_length = st.slider("PETAL_LENGTH", min_value=1.0, max_value=10.0, step=0.05)
petal_width = st.slider("PETAL_WIDTH", min_value=1.0, max_value=10.0, step=0.05)

if st.button("PREDICT"):
       features=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
       prediction=model.predict(features)
       if prediction[0]==0:
              st.markdown('<h3 style="color:blue;"> SETOSA </h3>',unsafe_allow_html=True)
       elif prediction[0]==1:
              st.markdown('<h3 style="color:blue;"> VERSICOLOR </h3>',unsafe_allow_html=True)
       else :
              st.markdown('<h3 style="color:blue;"> VARGINICA </h3>',unsafe_allow_html=True)
             
       
