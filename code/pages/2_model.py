import streamlit as st
import torch
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from streamlit_extras.row import row
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, T5Tokenizer, AutoModel, T5ForConditionalGeneration

if 'model' not in st.session_state:
    st.session_state['model'] = 0
if 'max_length' not in st.session_state:
    st.session_state['max_length'] = 0
if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 0
if 'repetition_penalty' not in st.session_state:
    st.session_state['repetition_penalty'] = 0

bitsandbyte_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16)

@st.cache_resource
def load_llm_model(max_length, temperature,  repetition_penalty):
    #this one is for running with GPU
    model = T5ForConditionalGeneration.from_pretrained('lmsys/fastchat-t5-3b-v1.0',
                                                       quantization_config = bitsandbyte_config,
                                                       device_map= 'auto')
    tokenizer =  AutoTokenizer.from_pretrained('lmsys/fastchat-t5-3b-v1.0')
    pipe = pipeline(
        task= 'text2text-generation', model=model, tokenizer=tokenizer, max_new_tokens=256, model_kwargs={"temperature":0,
                                                                                                         "repetition_penalty": 1.5}
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # llm = HuggingFacePipeline.from_model_id(model_id= 'lmsys/fastchat-t5-3b-v1.0', 
    #                                         task= 'text2text-generation',
    #                                         model_kwargs={ 
    #                                             # "device_map": "auto",
    #                                                     "max_length": 256, "temperature": 0,
    #                                                     "repetition_penalty": 1.5,
    #                                                      "quantization_config": bitsandbyte_config}) #add this quantization config
    
    
    # llm = HuggingFacePipeline.from_model_id(model_id= 'lmsys/fastchat-t5-3b-v1.0', 
    #                                     task= 'text2text-generation',
                                        
    #                                     model_kwargs={ "max_length": 256, "temperature": 0,
    #                                                   "torch_dtype":torch.float32,
    #                                                 "repetition_penalty": 1.3})
    return llm




st.title("Model Download")
# st.subheader("This page allows users to adjust some parameters of the model before downloading")
st.subheader("ผู้ใช้สามารถปรับเลือกการตั้งค่าต่อไปนี้เพื่อทำการดาวน์โลดโมเดล")

# model_row = row([2, 2, 2], vertical_align="bottom")
# max_length = model_row.number_input("max_length", value = 256)
# temperature = model_row.number_input("temperature", value = 0)
# repetition_penalty = model_row.number_input("repetition_penalty", value = 1.3)


max_length = st.number_input("max_length", value = 256, step = 128)
st.caption("""
           กำหนดจำนวนคำของโมเดลภาษา หากตั้งค่าน้อย โมเดลจะตอบสั้นและกระชับ ถ้าตั้งค่าให้มากๆ การตอบกลับอาจมีรายละเอียดมากขึ้น แต่ต้องระวังเพราะคำตอบที่ยาวเกินไปอาจไม่มีจุดโฟกัส
           """)
st.divider()

temperature = st.number_input("temperature", value = 0.0, step = 0.1, max_value = 1.0)
st.caption("""
           กำหนดความคิดสร้างสรรค์และความหลากหลายในการตอบของโมเดล ค่าที่ต่ำสุด 0 จะเป็นการตอบแบบมีการควบคุมสูงสุด โดย 1 จะมีความหลากหลายสูงสุด และสร้างสรรค์มากสุด แต่ความมีเหตุผลอาจลดลง
           """)
st.divider()
repetition_penalty = st.number_input("repetition_penalty", value = 1.3, step = 0.1, max_value = 2.0)
st.caption("""
           กำหนดให้โมเดลพยายามหลีกเลี่ยงการใช้คำหรือวลีเดียวกันซ้ำๆ ค่าที่สูงขึ้นจะทำให้โมเดลเลี่ยงการตอบโดยใช้คำ หรือวลีเดิมๆ
           """)

load_model_button = st.button("ดาวน์โลดโมเดล")

if load_model_button:
    st.session_state['max_length'] = max_length
    st.session_state['temperature'] = temperature
    st.session_state['repetition_penalty'] = repetition_penalty
    
    st.session_state['model'] = load_llm_model(max_length, temperature, repetition_penalty)
    
    
    st.write("⚠️ Please expect to wait **1 - 2 minutes **  for the application to download the 3-billion-parameter LLM")
    st.write('Successfully model loaded ✅')  
    # st.write('Successfully mผ te['repetition_penalty'])
    
    
# st.markdown(type(st.session_state['model']))