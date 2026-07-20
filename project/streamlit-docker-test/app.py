import streamlit as st

st.title("Hello Docker + Streamlit")

st.write("This is a test deployment on GCP Cloud Run.")

name = st.text_input("Your name")

if name:
    st.success(f"Hello, {name}!")