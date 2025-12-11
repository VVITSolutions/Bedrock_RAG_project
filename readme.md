# Chat with PDF using AWS Bedrock + LangChain + Streamlit

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS_Bedrock-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-10b981?style=for-the-badge" />
</p>

<p align="center">
  <strong>Ask questions to your PDFs using Claude 3 or Llama 3 on AWS Bedrock — fully local vector store with FAISS, beautiful Streamlit UI</strong>
</p>

<br>

<img width="1450" height="542" alt="image" src="https://github.com/user-attachments/assets/63f2a2b1-68d4-4f98-a8c2-1b9f227917a1" />


<br>

## Features

- Upload any number of PDFs (just drop them in `/data`)
- Automatic chunking + embedding with **Amazon Titan Embeddings**
- Lightning-fast similarity search using **FAISS**
- Answer questions using:
  - **Claude 3 Sonnet** (smartest)
  - **Llama 3 70B** (open weights)
- One-click vector store creation
- Clean Streamlit web interface
- 100% local vector index (no data leaves your machine except to Bedrock)

<br>

## Prerequisites

- Python 3.10–3.11
- AWS account with **Bedrock access enabled** (us-east-1 or us-west-2 recommended)
- IAM user/role with `bedrock:InvokeModel` and `bedrock-runtime` permissions

<br>

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/bedrock-pdf-chat.git
cd bedrock-pdf-chat

# 2. Create and activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your PDFs
#    → Create a folder called 'data' and put your PDF files inside
mkdir data
# copy your files into data/

# 5. Run the app

streamlit run pdf_app.py
