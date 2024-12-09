import streamlit as st
from dataset import load_dataset
from llm import chat_with_csv

st.set_page_config(layout='wide')
st.title("Onto-X llm")

# Load and display the dataset
data = load_dataset()
st.dataframe(data.head(), use_container_width=True)

# Input prompt 
st.info("Enter the Preferred Label of the entity you want to analyze")
input_text = st.text_area("Query")

if input_text:
    if st.button("Get Ancestors"):
        st.info("Your Query: " + input_text)
        result = chat_with_csv(data, input_text)

        # Result are displayed in a dictionary
        if isinstance(result, dict):
            st.json(result)
        else:
            st.warning("Received an unexpected result format. Displaying raw output:")
            st.write(result)
