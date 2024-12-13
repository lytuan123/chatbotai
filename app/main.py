# app/main.py
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import handle_query, init_system

app = FastAPI()

# Khởi tạo mô hình, FAISS index, chain... khi server start
@app.on_event("startup")
async def startup_event():
    init_system()  # Hàm này load FAISS index, LLM, chain, cache...

class QueryRequest(BaseModel):
    query: str
    topic: str

@app.post("/ask")
def ask(req: QueryRequest):
    # topic: "vondautu" hoặc "xaydung"
    # query: câu hỏi người dùng
    try:
        answer = handle_query(req.query, req.topic)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chạy app bằng:
# uvicorn app.main:app --host 0.0.0.0 --port 8000
