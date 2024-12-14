from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

from utils import chatbot_manager, ChatRequest

app = FastAPI(
    title="PDF Chatbot API",
    description="A chatbot API that can answer questions based on PDF documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatInput(BaseModel):
    query: str
    topic: str
    chat_history: List[Dict[str, str]] = []

@app.post("/chat")
async def chat_endpoint(input: ChatInput):
    try:
        chat_request = ChatRequest(
            query=input.query,
            topic=input.topic,
            chat_history=input.chat_history
        )
        result = chatbot_manager.handle_query(chat_request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topics")
async def get_available_topics():
    return list(chatbot_manager.chains.keys())

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Optional: Swagger UI
@app.get("/")
async def root():
    return {"message": "PDF Chatbot API is running"}
