# MediBot — Medical Reference Chatbot

A retrieval-augmented generation (RAG) chatbot that answers medical questions grounded in the *Gale Encyclopedia of Medicine*. Instead of relying on an LLM's parametric memory — which hallucinates confidently on clinical detail — every answer is retrieved from the source text and cited back to it.

Built with LangChain, Pinecone, and Flask. Deployed to AWS EC2 via a GitHub Actions CI/CD pipeline.

---

## Why RAG for this

An LLM asked "what is acromegaly?" will produce a fluent answer from training data with no way to verify where it came from. For medical content, that's the wrong failure mode.

This pipeline instead:

1. Chunks the Gale Encyclopedia into ~500-character passages
2. Embeds each chunk into a 384-dimensional vector using `all-MiniLM-L6-v2`
3. Stores those vectors in a Pinecone index
4. At query time, embeds the question, retrieves the 3 nearest chunks by cosine similarity, and passes **only those chunks** to the LLM as context

The LLM's job is reduced from *recall* to *summarization over retrieved evidence* — a much narrower task, and one where the source of every claim is traceable.

---

## Architecture

```
PDF (Gale Encyclopedia)
   │
   ├─ PyPDFLoader ──────────► raw documents
   │
   ├─ RecursiveCharacterTextSplitter (500 chars, 20 overlap)
   │                        ► text chunks
   │
   ├─ all-MiniLM-L6-v2 ─────► 384-d vectors
   │
   └─ Pinecone (cosine) ────► persistent index
                                    │
User question ──► MiniLM ──► similarity search (k=3)
                                    │
                              retrieved chunks
                                    │
                              GPT-4o-mini ──► grounded answer
```

**Why MiniLM:** 22M parameters, runs on CPU in milliseconds, 384 dimensions keeps the index small and search fast. The tradeoff is less nuance than a larger embedding model — acceptable here, and it means no GPU is needed to index thousands of chunks.

**Critical constraint:** the Pinecone index dimension (384) must match the embedding model's output. Most Pinecone tutorials use 1536 because they assume OpenAI embeddings. A mismatch here fails silently or obscurely.

---

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector store | Pinecone (serverless, cosine metric) |
| LLM | OpenAI `gpt-4o-mini` |
| Backend | Flask |
| Deployment | Docker → AWS ECR → EC2 |
| CI/CD | GitHub Actions (self-hosted runner on EC2) |

---

## Running locally

**1. Clone and create the environment**

```bash
git clone https://github.com/TariqHusainKhan/medical_chatBot.git
cd medical_chatBot

conda create -n medibot python=3.11 -y
conda activate medibot
```

> Python 3.11, not 3.10 — some dependencies (`pydantic-core`) have no working wheels on newer or older interpreters.

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add credentials**

Create a `.env` file in the project root:

```
PINECONE_API_KEY=pcsk_...
OPENAI_API_KEY=sk-proj-...
```

No quotes, no trailing whitespace. `.env` is gitignored — never commit it.

**4. Add the source PDF**

Place the Gale Encyclopedia PDF in `data/`.

**5. Build the vector index**

```bash
python store_index.py
```

This loads the PDF, chunks it, embeds every chunk on CPU, and upserts to Pinecone. Expect several minutes with no progress output — it's working.

**6. Launch**

```bash
python app.py
```

Open **http://localhost:5000**.

---

## Project structure

```
medical_chatBot/
├── app.py              # Flask server + RAG chain assembly
├── store_index.py      # One-time: PDF → chunks → embeddings → Pinecone
├── src/
│   ├── helper.py       # Loading, chunking, embedding functions
│   └── prompt.py       # System prompt
├── templates/
│   └── chat.html       # Frontend
├── data/               # Source PDF (gitignored)
├── research/
│   └── trials.ipynb    # Development notebook
├── requirements.txt
├── Dockerfile
└── .env                # Credentials (gitignored)
```

`store_index.py` runs **once** to populate Pinecone. `app.py` connects to the existing index via `from_existing_index` — it does not re-embed on startup.

---

## Deployment: AWS EC2 with GitHub Actions

The pipeline builds a Docker image on every push to `main`, pushes it to ECR, and the EC2 instance — registered as a self-hosted runner — pulls and runs it.

### 1. IAM user

Create an IAM user for deployment with programmatic access and these policies:

- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

Save the access key ID and secret.

### 2. ECR repository

Create a repository named `medicalchatbot` and note its URI:

```
<your-account-id>.dkr.ecr.ap-south-1.amazonaws.com/medicalchatbot
```

### 3. EC2 instance

Launch an Ubuntu instance (`t2.micro` is free-tier eligible). In the security group, allow inbound on **22** (SSH, your IP) and **5000** (HTTP, anywhere).

SSH in and install Docker:

```bash
sudo apt-get update -y
sudo apt-get upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### 4. Self-hosted runner

In your GitHub repo: **Settings → Actions → Runners → New self-hosted runner**. Select Linux, then run the provided commands on the EC2 instance in order.

### 5. Repository secrets

**Settings → Secrets and variables → Actions**, add:

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | From the IAM user |
| `AWS_SECRET_ACCESS_KEY` | From the IAM user |
| `AWS_DEFAULT_REGION` | `ap-south-1` |
| `ECR_REPO` | Your ECR URI |
| `PINECONE_API_KEY` | Pinecone key |
| `OPENAI_API_KEY` | OpenAI key |

Push to `main` and the workflow builds, pushes, and deploys.

> **Cost note:** terminate the EC2 instance when you're done — *terminate*, not *stop*. A stopped instance still holds EBS storage and continues to bill.

---

## Known limitations

- **Embedding quality.** MiniLM occasionally conflates closely-related conditions (e.g. distinguishing type 1 from type 2 diabetes context). A larger model like `bge-large` would improve retrieval precision at the cost of index size and CPU time.
- **No conversation memory.** Each question is answered independently. Follow-ups like "and what causes it?" have no antecedent.
- **Fixed `k=3`.** Retrieval always pulls three chunks regardless of question complexity. A reranker or dynamic `k` would help on broad questions.
- **Single-source corpus.** Everything is grounded in one encyclopedia. It cannot answer anything the Gale Encyclopedia doesn't cover, and correctly says so.

---

## Disclaimer

This is a reference and study tool built to demonstrate a RAG pipeline. It is not a diagnostic instrument and is not a substitute for a qualified clinician.