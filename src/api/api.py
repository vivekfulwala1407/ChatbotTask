from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ..chatbot.rag_engine import answer_query

app = FastAPI(title="RAG Chatbot")

class ChatRequest(BaseModel):
    question: str
    k: int = 5

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    try:
        res = answer_query(req.question)
        
        return {"answer": res if isinstance(res, str) else res.get("answer", "I don't know / not available in the dataset")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))