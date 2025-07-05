# Multi PDF Question Answering System using LLAMA 3.3 and GROQ Cloud

## Overview

This is a Python-based PDF Question Answering (QA) system that allows users to upload multiple PDF files and ask questions about their contents. The application uses advanced natural language processing techniques to extract and retrieve relevant information from the uploaded documents.

## Features

- Multiple PDF file upload
- Document text extraction and chunking
- Vector-based semantic search
- Question answering using state-of-the-art language model
- Gradio-based web interface for easy interaction

## Technologies Used

- Gradio: Web interface creation
- LangChain: Document processing and QA workflow
- HuggingFace: Embedding generation
- Groq: Large Language Model (LLM) for question answering
- ChromaDB: Vector store for document embeddings

## Prerequisites

- Python 3.12.10
- pip (Python package manager)

## Installation



1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
- Create a `.env` file in the project root
- Add GROQ API keys

## Usage

Run the application:
```bash
python main.py
```

1. Upload PDF files using the file upload interface
2. Click "Process PDFs" to prepare the documents
3. Type your question in the query input
4. Click "Submit Query" to get answers with source document references

## How It Works

1. PDF documents are loaded and split into smaller chunks
2. Document chunks are converted to vector embeddings
3. When a query is submitted, semantic similarity search finds relevant document chunks
4. The language model generates an answer based on the retrieved chunks

## Configuration

- Adjust `chunk_size` and `chunk_overlap` in `split_documents()` to fine-tune document processing
- Change the embedding model in `create_vector_store()` 
- Modify Groq LLM parameters in `create_qa_chain()`

## Dependencies

- gradio
- python-dotenv
- langchain
- transformers
- chromadb
- groq

## Limitations

- Performance depends on the quality and complexity of input PDFs
- Large documents may require more computational resources
- Accuracy varies based on the chosen embedding and language models

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the GitHub repository.



## Acknowledgments

- LangChain
- HuggingFace
- Groq
- Gradio



…or create a new repository on the command line
echo "# TotaPakhiV2" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/hey-ashik/TotaPakhiV2.git
git push -u origin main
…or push an existing repository from the command line
git remote add origin https://github.com/hey-ashik/TotaPakhiV2.git
git branch -M main
git push -u origin main




# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 799840519333.dkr.ecr.ap-south-1.amazonaws.com/totapakhiibot

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
   - ECR_REPO
   - PINECONE_API_KEY
   - OPENAI_API_KEY