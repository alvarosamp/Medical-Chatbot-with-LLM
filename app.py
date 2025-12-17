from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

# Hugging Face (API nova)
from huggingface_hub import InferenceClient

# Pinecone + embeddings
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt


# =======================
# App & Env
# =======================
app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL_REPO = os.getenv(
    "HF_MODEL_REPO",
    "mistralai/Mistral-7B-Instruct-v0.2"
)

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# =======================
# Hugging Face Client
# =======================
hf_client = InferenceClient(
    model=HF_MODEL_REPO,
    token=HUGGINGFACEHUB_API_TOKEN,
)


# =======================
# Embeddings & Vector DB
# =======================
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# =======================
# RAG Manual (CORE)
# =======================
def generate_answer(question: str) -> str:
    # 1. Recuperar contexto
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 2. Montar mensagens (chat-based)
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    # 3. Chamada ao modelo (API NOVA HF)
    response = hf_client.chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.1,
    )

    return response.choices[0].message.content


# =======================
# Routes
# =======================
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    return generate_answer(msg)


# =======================
# Run
# =======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
