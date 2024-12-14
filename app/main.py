from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.chatbot import handle_query

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API"}

@app.post("/chat")
async def chat(query: str, selected_menu: str):
    try:
        chat_history, response = handle_query(query, selected_menu, [])
        return JSONResponse(content={"chat_history": chat_history, "response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
