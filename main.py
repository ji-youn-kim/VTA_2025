import streamlit as st

from first_page import *
from second_page import *
import os

os.environ["LANGSMITH_TRACING"]='true'
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]=st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"]="AI504_2025"

def main():
    if "student_id" not in st.session_state:
        first_page()
    else:
        second_page()

if __name__ == "__main__":
    main()