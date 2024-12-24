import os
from pydantic import BaseModel
from tokenizers import Tokenizer
from fastapi import (APIRouter, 
                     HTTPException)
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.config import settings


router = APIRouter()


class DocQuery(BaseModel):
    query: str
    file_path: str

@router.post('/query')
async def read_doc(req: DocQuery):
    if not (os.path.isfile(req.file_path) and req.file_path.lower().endswith('.pdf')):
        raise HTTPException(status_code=400, detail='Invalid file path.')
    try:
        loader = PyPDFLoader(req.file_path)
        docs = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)

        # Embedding model
        embeddings = MistralAIEmbeddings(
            model="mistral-embed", 
            mistral_api_key=settings.MISTRAL_API_KEY,
            tokenizer=Tokenizer.from_pretrained('mistralai/Mixtral-8x7B-v0.1', token=settings.HF_TOKEN)
        )

        # Create the vector store and a retriever interface
        vector = FAISS.from_documents(documents, embeddings)
        retriever = vector.as_retriever(search_type="similarity", k=5)

        # LLM
        model = ChatMistralAI(model_name='open-mistral-7b', mistral_api_key=settings.MISTRAL_API_KEY)

        # Prompt template
        prompt = ChatPromptTemplate.from_template(
            """Answer the following question based only on the provided context:

            <context>
            {context}
            </context>

            Question: {input}"""
        )

        # Create a retrieval chain to answer the queries
        document_chain = create_stuff_documents_chain(model, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": req.query})

        return response
    
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
