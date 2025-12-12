from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    question: str
    k: Optional[int] = 5

class ChatResponse(BaseModel):
    answer: str
    evidence: list
    filters: dict