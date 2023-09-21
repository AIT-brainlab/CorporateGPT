FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# https://vsupalov.com/docker-arg-env-variable-guide/
# https://bobcares.com/blog/debian_frontendnoninteractive-docker/
ARG DEBIAN_FRONTEND=noninteractive
# Timezone
ENV TZ="Asia/Bangkok"

RUN apt update && apt upgrade -y
# Set timezone
RUN apt install -y tzdata
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# Set locales
# https://leimao.github.io/blog/Docker-Locale/
RUN apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LC_ALL en_US.UTF-8 
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en

RUN apt install -y python3 python3-pip

RUN pip3 install unstructured
RUN pip3 install python-pptx
RUN pip3 install googletrans==3.1.0a0
RUN pip3 install streamlit==1.24.0
RUN pip3 install streamlit_extras
RUN pip3 install PyPDF2
RUN pip3 install torch
RUN pip3 install streamlit-chat
RUN pip3 install gspread
RUN pip3 install langchain
RUN pip3 install InstructorEmbedding
RUN pip3 install sentence_transformers
RUN pip3 install faiss-cpu

# Clean apt 
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /root/code

COPY ./code /root/code/
CMD tail -f /dev/null