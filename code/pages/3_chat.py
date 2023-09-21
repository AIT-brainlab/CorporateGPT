import streamlit as st
from streamlit_chat import message as st_message
import pandas as pd
import numpy as np
import datetime
import gspread
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter

from googletrans import Translator

# from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings


from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory


from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT


prompt_template = """
You are the chatbot and your job is to give answers.
MUST only use the following pieces of context to answer the question at the end. If the answers are not in the context or you are not sure of the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
When encountering abusive, offensive, or harmful language, such as fuck, bitch,etc,  just politely ask the users to maintain appropriate behaviours.
Always make sure to elaborate your response.
Never answer with any unfinished response
Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
# chain_type_kwargs = {"prompt": PROMPT}

@st.cache_resource
def load_conversational_qa_memory_retriever():

    question_generator = LLMChain(llm=llm_model, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(llm_model, chain_type="stuff", prompt = PROMPT)
    memory = ConversationBufferWindowMemory(k = 3,  memory_key="chat_history", return_messages=True,  output_key='answer')
    
    
    
    conversational_qa_memory_retriever = ConversationalRetrievalChain(
        retriever=vector_database.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
        memory = memory,
        get_chat_history=lambda h :h)
    return conversational_qa_memory_retriever, question_generator

def new_retrieve_answer():
    translated_to_eng = thai_to_eng(st.session_state.my_text_input).text 
    prompt_answer=  translated_to_eng + ". Try to be elaborate and informative in your answer."
    answer = conversational_qa_memory_retriever({"question": prompt_answer })

    print(f"condensed quesion : {question_generator.run({'chat_history': answer['chat_history'], 'question' : prompt_answer})}")

    print(answer["chat_history"])
    
    st.session_state.chat_history.append({"message": st.session_state.my_text_input, "is_user": True})
    st.session_state.chat_history.append({"message": eng_to_thai(answer['answer'][6:]).text , "is_user": False})

    st.session_state.my_text_input = ""

    return eng_to_thai(answer['answer']).text #this positional slicing helps remove "<pad> " at the beginning
    
def clean_chat_history():
    st.session_state.chat_history = []
    conversational_qa_memory_retriever.memory.chat_memory.clear() #add this to remove

def thai_to_eng(text):
    translated = translator.translate(text, src='th', dest ='en')
    return translated

def eng_to_thai(text):
    translated = translator.translate(text, src='en', dest ='th')
    return translated

if "history" not in st.session_state: #this one is for the google sheet logging
    st.session_state.history = []


if "chat_history" not in st.session_state: #this one is to pass previous messages into chat flow
    st.session_state.chat_history = []
    


llm_model =  st.session_state['model']
vector_database =  st.session_state['faiss_db']
conversational_qa_memory_retriever, question_generator = load_conversational_qa_memory_retriever()
translator = Translator()


print("all load done")


# Try adding this to set to clear the memory in each session
if st.session_state.chat_history == []:
    conversational_qa_memory_retriever.memory.chat_memory.clear()



st.write("# extraGPT 🤖 ")

with st.expander("key information"):
    st.write(  st.session_state['chunked_df'], unsafe_allow_html=True)
    st.markdown(st.session_state['max_length'])
    st.markdown(st.session_state['temperature'])
    st.markdown(st.session_state['repetition_penalty'])



st.write(""" ⚠️ 
การถาม 1 คำถามอาจใช้เวล ~ 10 - 20 วินาที หรือมากกว่า เนื่องจากแอปนี้กำลังทำงานบน CPU สำหรับ LLM ที่มีพารามิเตอร์ 3 พันล้านตัว
หากต้องการให้เร็วขึ้นจำเป็นที่จะต้องใช้ GPU เข้ามาช่วย
""")

st.markdown("---")
st.write(" ")
st.write("""
         ### ❔ Ask a question
         """)




for chat in st.session_state.chat_history:
    st_message(**chat)

query_input = st.text_input(label= 'พิมพ์คำถามที่นี้ แล้วกด "enter"' , key = 'my_text_input', on_change= new_retrieve_answer )



clear_button = st.button("เริ่มบทสนทนาใหม่",
                         on_click=clean_chat_history)
