# Retrieval Augmented Generation based Application using FastAPI and LangChain

This project focuses on developing an application using the Retrieval-Augmented Generation (RAG) framework combined with FastAPI. The application is designed to handle user queries by leveraging the content of a provided PDF file. By accepting a file path as input, it processes and extracts relevant information from the PDF to generate accurate and contextually relevant answers.

## Setup Guide
### Prerequisites
- Python 3.10+
- [Mistral API Access Key](https://console.mistral.ai/api-keys/)
- [HuggingFace API Access Token](https://huggingface.co/settings/tokens)

### Installation Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/suhail-chand/rag-fastapi.git
   cd rag-fastapi
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a .env file and add the following variables**:
   ```bash
   HF_TOKEN='<HuggingFace_Access_Token>'
   MISTRAL_API_KEY='<Mistral_API_Key>'
   ```

5. **Run the FastAPI server**:
   ```bash
   fastapi dev .\src\main.py
   ```

FastAPI application will be running at `http://127.0.0.1:8000` and to access the Swagger UI navigate to `http://127.0.0.1:8000/docs`.
