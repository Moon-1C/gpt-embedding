# main.py
from fastapi import FastAPI
from pydantic import BaseModel
# 머신러닝 라이브러리와 모델을 로드하는 코드를 추가합니다.
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from typing import Optional
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
app = FastAPI()

import os
#@markdown https://platform.openai.com/account/api-keys
OPENAI_API_KEY = ""

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

import pandas as pd

df = pd.read_excel("완성본_태그추가.xlsx")

from langchain.document_loaders import DataFrameLoader

loader = DataFrameLoader(df, page_content_column="name")


class InputData(BaseModel):
    data: str

@app.post("/predict")
def predict(input_data:InputData):
    # 머신러닝 모델을 사용하여 예측을 수행하는 코드를 추가합니다.
    
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceEmbeddings()
    
    chat = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.9)
    # index = VectorstoreIndexCreator(
    # vectorstore_cls=FAISS,
    # embedding=embeddings,
    # ).from_loaders([loader])

    # 파일로 저장
    # index.vectorstore.save_local("faiss-nj")
    # sys = SystemMessage(content="당신은 제주도 여행지를 추천을 해주는 전문 AI입니다. 이때 추천지는 단어로만 말하고 ','로 구분합니다.")
    # chat = chat([sys])

    fdb = FAISS.load_local("faiss-nj", embeddings)
    index = VectorStoreIndexWrapper(vectorstore=fdb)
    

    return index.query(input_data.data, llm=chat)