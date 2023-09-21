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
from url_list import url_list

import os
import glob
from PyPDF2 import PdfReader

if 'faiss_db' not in st.session_state:
    st.session_state['faiss_db'] = 0
    
if 'chunked_count_list' not in st.session_state:
    st.session_state['chunked_count_list'] = 0
    
if 'chunked_df' not in st.session_state:
    st.session_state['chunked_df'] = 0


def make_clickable(link):
    text = link.split()[0]
    return f'<a target="_blank" href="{link}">{text}</a>'



user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36'
headers = {'User-Agent': user_agent}



def scrape_url(url_list):
    all_whole_text = []
    for url in url_list:
        main_url = url
        html_doc = requests.get(main_url, headers =headers )
        soup = BeautifulSoup(html_doc.text, 'html.parser')
        whole_text = ""
        
        for paragraph in soup.find_all():
            if paragraph.name in ["p", "ol"]:
                whole_text += paragraph.text.replace('\xa0', '').replace('\n', '').strip()
                
        all_whole_text.append(whole_text)
    
    return all_whole_text

def create_count_list(chuked_text):
    original_count_list = []

    for item in range(len(chuked_text)):
        original_count_list.append(chuked_text[item].metadata['document'])
    item_counts = Counter(original_count_list)
    count_list = list(item_counts.values())
    return count_list

def thai_to_eng(text):
    translated = translator.translate(text, src='th', dest ='en')
    return translated

def eng_to_thai(text):
    translated = translator.translate(text, src='en', dest ='th')
    return translated



#function to manage pdf
def read_pdf_text(pdf_path):
    pdf_pattern = os.path.join(pdf_path, '*.pdf')
    pdf_files = glob.glob(pdf_pattern)
    
    all_text  = []
    all_pages = []
    for file in pdf_files:
        
        # creating a pdf reader object
        reader = PdfReader(file)
        all_pages.append(len(reader.pages))
        page_text = ""
        for page in range(len(reader.pages)):
            page_text += reader.pages[page].extract_text()
        all_text.append(page_text)
        
    pdf_metadatas = [{"document": i, "filename" : j[11:]} for i, j in enumerate(pdf_files)]
    
    return pdf_files, all_text, all_pages,  pdf_metadatas



# Replace 'path_to_folder' with the path of the folder containing your PDF files
path_to_folder_pdf = 'pdf_folder'
pdf_files, pdf_text_list, pdf_all_pages, pdf_metadatas= read_pdf_text(path_to_folder_pdf)
# st.set_page_config(page_title=None, page_icon=None, layout="wide")

# url_list =  ["https://www.mindphp.com/คู่มือ/openerp-manual.html#google_vignette", 
#                 "https://www.mindphp.com/คู่มือ/openerp-manual/7874-refund.html",
#                 "https://www.mindphp.com/คู่มือ/openerp-manual/8842-50-percent-discount-on-erp.html",
#                 "https://www.mindphp.com/คู่มือ/openerp-manual/7873-hr-payroll-account.html",
#                 "https://www.mindphp.com/คู่มือ/openerp-manual/4255-supplier-payments.html"]#or whatever default

metadatas = [{"document": i, "url" : j} for i, j in enumerate(url_list)]

scrape_list = scrape_url(url_list)
translator = Translator()


st.title("Data Chunking")
# st.subheader("The main purpose of this page is to split the entired scrape data into small chunks for more finegrained knowledge retrieval")
st.subheader("จุดประสงค์หลักของหน้านี้คือแบ่งข้อมูลที่ถูกสกัดมาทั้งหมดเป็นชิ้นย่อยๆ เพื่อช่วยให้การรียกข้อมูลละเอียดมากขึ้น")

var1 = st.number_input("Chunk Size", value = 1200, step= 100)
# st.caption("Chunk size determines the number of characters remaining in each document after chunking")
st.caption("Chunk size กำหนดจำนวนอักขระที่เหลือในแต่ละเอกสารหลังจากการแบ่งชิ้น")

st.divider()

var2 = st.number_input("Chunk Overlap Size", value = 100, step= 10)
# st.caption("Chunk overlap size determines the number of characters overlapping between 2 adjacent documents")
st.caption("Chunk overlap size กำหนดจำนวนอักขระที่ซ้อนทับกันระหว่างเอกสารที่อยู่ติดกัน 2 ชิ้น")





split_button = st.button("เริ่มแบ่งข้อมูล")

if split_button:
    
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = var1,
        chunk_overlap  = var2,
        length_function = len
    )

    #chunk url
    chuked_text = text_splitter.create_documents([doc for doc in scrape_list], metadatas = metadatas)
    chunked_count_list = create_count_list(chuked_text)

    print(len(url_list), len(chunked_count_list))

    url_dataframe = pd.DataFrame({'link': url_list, 'number_of_chunks': chunked_count_list})
    url_dataframe['link'] = url_dataframe['link'].apply(make_clickable)
    url_dataframe = url_dataframe.to_html(escape=False)

    st.session_state['chunked_df'] = url_dataframe

    #chunk pdf
    pdf_chuked_text = text_splitter.create_documents([doc for doc in pdf_text_list], metadatas = pdf_metadatas)
    pdf_chunked_count_list = create_count_list(pdf_chuked_text)
    
    pdf_url_dataframe = pd.DataFrame({'pdf_name': pdf_files, 
                                      'number_of_pages': pdf_all_pages,
                                      'number_of_chunks': pdf_chunked_count_list})
   
   
    # st.dataframe(url_dataframe)
    
    # with st.expander("chunked items"):    
    #     st.json(chuked_text)
    
    
    
    translated_chunk_text = copy.deepcopy(chuked_text)
    for chunk in range(len(translated_chunk_text)):
        translated_chunk_text[chunk].page_content = thai_to_eng(translated_chunk_text[chunk].page_content).text
        
    
    pdf_translated_chunk_text = copy.deepcopy(pdf_chuked_text)
    for chunk in range(len(pdf_translated_chunk_text)):
        pdf_translated_chunk_text[chunk].page_content = thai_to_eng(pdf_translated_chunk_text[chunk].page_content).text
        

    translated_chunk_text.extend(pdf_translated_chunk_text)
    
    

    embedding_model = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-base',
                                                    model_kwargs = {'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')})

    faiss_db = FAISS.from_documents(translated_chunk_text, embedding_model)


    st.session_state['faiss_db'] = faiss_db
    

    st.session_state['chunked_count_list'] = chunked_count_list
        
    st.divider()
    st.header("Data Summary")
    st.subheader("URL sources")
    st.write(url_dataframe, unsafe_allow_html=True)
    st.write('\n')
    st.write("มีจำนวนลิ้งค์ทั้งหมด", len(url_list), " ลิ้ง โดยมีจำนวนเอกสารหลังการแบ่งทั้งสิ้น ", len(translated_chunk_text), "เอกสาร")
    st.write('\n')
    st.write('\n')
    st.subheader("PDF sources")
    st.write(pdf_url_dataframe, unsafe_allow_html=True)
    st.write('\n')
    st.write("มีจำนวนไฟล์ทั้งหมด ", len(pdf_files), " ไฟล์ โดยมีจำนวนเอกสารหลังการแบ่งทั้งสิ้น ", len(pdf_translated_chunk_text), "เอกสาร")
    
    st.write('Successfully preprocessed data ✅ Please go the model page') 

    




