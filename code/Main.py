import streamlit as st
import pandas as pd
import copy
from googletrans import Translator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_extras.row import row
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from collections import Counter
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from collections import Counter

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

# st.markdown("main page")    


col1, col2, col3 = st.columns([0.2, 0.5, 0.2])
col2.image("https://cdn-icons-png.flaticon.com/512/2040/2040946.png", use_column_width=True )


row2_col1, row2_col2, row2_col3 = st.columns([0.2, 0.5, 0.2])
row2_col2.markdown("<h3 style='text-align: center; color: black;'>Chatbot for Thai Corporate Document Q&A</h3>", unsafe_allow_html=True )




# st.markdown("Please start customizing dataset and downloading the model on the next pages")


st.markdown("<p style='text-align: center; color: black;'>Please start customizing dataset and downloading the model on the next pages</p>", unsafe_allow_html=True)