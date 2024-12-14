import os
import logging
import json
import hashlib
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from groq import Groq

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load biến môi trường
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    FAISS_INDEX_DIR = "faiss_index"
    CACHE_FILE = "cache.json"
    MODEL_NAME = 'intfloat/multilingual-e5-large'
    
    FAISS_INDEXES = {
        "vondautu": os.path.join(FAISS_INDEX_DIR, "vondautu_index"),
        "xaydung": os.path.join(FAISS_INDEX_DIR, "xaydung_index"),
    }

class GroqLLMService:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate_response(self, prompt: str, max_tokens: int = 1024):
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"LLM Generation Error: {e}")
            return "Xin lỗi, đã xảy ra lỗi trong quá trình xử lý."

class VectorDBService:
    def __init__(self, model_name: str):
        self.embedding_model = SentenceTransformerEmbeddings(
            model_name=model_name, 
            model_kwargs={"device": "cpu"}
        )
        self.indices = self._load_indices()

    def _load_indices(self):
        indices = {}
        for menu_item, index_dir in Config.FAISS_INDEXES.items():
            try:
                indices[menu_item] = FAISS.load_local(
                    index_dir, 
                    self.embedding_model, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                logging.error(f"Failed to load index {menu_item}: {e}")
        return indices

    def get_retriever(self, topic):
        return self.indices.get(topic).as_retriever() if topic in self.indices else None

class ChatService:
    def __init__(self, llm_service: GroqLLMService, vector_db_service: VectorDBService):
        self.llm_service = llm_service
        self.vector_db_service = vector_db_service
        self.cache = self._load_cache()

    def _load_cache(self):
        try:
            with open(Config.CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save_cache(self):
        with open(Config.CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4)

    def handle_query(self, query: str, selected_menu: str, chat_history: List[Dict[str, str]]):
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        
        if query_hash in self.cache:
            return self.cache[query_hash]

        retriever = self.vector_db_service.get_retriever(selected_menu)
        if not retriever:
            return "Không tìm thấy chủ đề phù hợp."

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm_service.client,
            retriever=retriever,
            memory=memory,
            verbose=False
        )

        result = chain({"question": query})
        answer = result["answer"]
        
        self.cache[query_hash] = answer
        self._save_cache()

        return answer

# FastAPI Application
app = FastAPI(title="PDF Chatbot API")

# Initialize services
llm_service = GroqLLMService(Config.GROQ_API_KEY)
vector_db_service = VectorDBService(Config.MODEL_NAME)
chat_service = ChatService(llm_service, vector_db_service)

# Endpoint
@app.post("/chat")
async def chat_endpoint(
    query: str, 
    selected_menu: str, 
    chat_history: Optional[List[Dict[str, str]]] = None
):
    try:
        response = chat_service.handle_query(query, selected_menu, chat_history or [])
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
