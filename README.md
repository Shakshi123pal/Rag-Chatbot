# 🤖 Chat with multiple PDFs locally
I built a RAG (Retrieval-Augmented Generation) chatbot that can ingest multiple PDFs, perform semantic search, and generate context-grounded answers to user questions. I implemented the embedding and similarity retrieval flow, designed the UI using Gradio, and automated deployment using Docker and scripts. This project demonstrates my practical experience with real-world NLP pipelines and model integration.
<img width="1919" height="1021" alt="image" src="https://github.com/user-attachments/assets/9da44c2c-fd29-4f60-8a8b-fe9e6bb0f859" />


[`Todo`](#🎯-todo)

# ⭐️ Key Features

- Easy to run on `Local` or `Kaggle` (new)
- Using any model from `Huggingface` and `Ollama`
- Process multiple PDF inputs.
- Chat with multiples languages (Coming soon).
- Simple UI with `Gradio`.

# 💡 Idea (Experiment)

![](./assets/rag-flow.svg)

![](./assets/retriever.svg)

# 💻 Setup

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

#### 2.2.3 Install manually

##### 1. `Ollama`

- MacOS, Window: [Download](https://ollama.com/)

- Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

##### 2. `Ngrok`

- Macos

```bash
brew install ngrok/ngrok/ngrok
```

- Linux

```bash
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
| sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
&& echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
| sudo tee /etc/apt/sources.list.d/ngrok.list \
&& sudo apt update \
&& sudo apt install ngrok
```

##### 3. Install `rag_chatbot` Package

```bash
source ./scripts/install.sh
```

### 2.3 Run

```bash
source ./scripts/run.sh
```

or

```bash
python -m rag_chatbot --host localhost
```

- Using Ngrok

```bash
source ./scripts/run.sh --ngrok
```

### 3. Go to: `http://0.0.0.0:7860/` or Ngrok link after setup completed

## 🎯 Todo

- [x] Add evaluation.
- [x] Better Document Processing.
- [ ] Support better Embedding Model for Vietnamese and other languages.
- [ ] ReAct Agent.
- [ ] Document mangement (Qrdant, MongoDB,...)


