# 🤖 Chat with multiple PDFs locally
I built a RAG (Retrieval-Augmented Generation) chatbot that can ingest  PDFs, perform semantic search, and generate context-grounded answers to user questions. I implemented the embedding and similarity retrieval flow, designed the UI using Gradio, and automated deployment using Docker and scripts. This project demonstrates my practical experience with real-world NLP pipelines and model integration.
<img width="1919" height="1021" alt="image" src="https://github.com/user-attachments/assets/9da44c2c-fd29-4f60-8a8b-fe9e6bb0f859" />




# 💡 Idea (Experiment)

![](./assets/rag-flow.svg)



## 1. Kaggle (Recommended)

- Import [`notebooks/kaggle.ipynb`](notebooks/kaggle.ipynb) to Kaggle
- Replace `<YOUR_NGROK_TOKEN>` with your tokens.

## 2. Local

### 2.1. Clone project

```bash
git clone https://github.com/Shakshi123pal/Rag-Chatbot.git
cd Rag-Chatbot
```

### 2.2 Install

#### 2.2.1 Docker

```bash
docker compose up --build
```

#### 2.2.2 Using script (Ollama, Ngrok, python package)

```bash
source ./scripts/install_extra.sh
```


```

##### 3. Install `rag_chatbot` Package

```bash
source ./scripts/install.sh
```

### 2.3 Run



python -m rag_chatbot --host localhost
```



### 3. Go to: `http://0.0.0.0:7860/` 



