import os
import shutil
import tempfile
from pydantic import BaseModel
from tokenizers import Tokenizer
from fastapi import (File, 
                     UploadFile)
from fastapi import (APIRouter, 
                     HTTPException)
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.document_loaders import (PyPDFLoader, 
                                                  UnstructuredWordDocumentLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.config import settings


router = APIRouter()

class Query(BaseModel):
    query: str


@router.post('/store')
async def store_document(file: UploadFile = File(...)):
    """
    Upload and store PDF or Word document content into FAISS vector store.
    """
    if not (file.filename.lower().endswith('.pdf') or file.filename.lower().endswith('.docx')):
        raise HTTPException(status_code=400, detail='Only PDF and DOCX files are accepted.')

    try:
        # Save uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, file.filename)

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load document based on file type
        if file.filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
        elif file.filename.lower().endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(temp_file_path)

        docs = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)

        # Embeddings
        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=settings.MISTRAL_API_KEY,
            tokenizer=Tokenizer.from_pretrained('mistralai/Mixtral-8x7B-v0.1', token=settings.HF_TOKEN)
        )

        if os.path.exists(settings.FAISS_STORE_PATH):
            # Load existing FAISS index and add new documents
            vector_store = FAISS.load_local(settings.FAISS_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            vector_store.add_documents(documents)
        else:
            # Create new FAISS index
            vector_store = FAISS.from_documents(documents, embeddings)

        # Save locally
        vector_store.save_local(settings.FAISS_STORE_PATH)

        return {"message": "Document content stored successfully."}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
        

@router.post('/query')
async def query_doc(req: Query):
    """
    Query LLM with the context provided by the uploaded documents.
    """
    try:
        # Embedding model
        embeddings = MistralAIEmbeddings(
            model="mistral-embed", 
            mistral_api_key=settings.MISTRAL_API_KEY,
            tokenizer=Tokenizer.from_pretrained('mistralai/Mixtral-8x7B-v0.1', token=settings.HF_TOKEN)
        )

        # Load the vector store and a retriever interface
        vector_store = FAISS.load_local(settings.FAISS_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_type="similarity", k=5)

        # LLM
        model = ChatMistralAI(model_name='open-mistral-7b', mistral_api_key=settings.MISTRAL_API_KEY)

        # Prompt template
        prompt = ChatPromptTemplate.from_template(
            """Answer the following question based on the provided context:

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
