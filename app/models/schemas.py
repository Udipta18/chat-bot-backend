from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []

class TranscriptionResponse(BaseModel):
    text: str

class ErrorResponse(BaseModel):
    detail: str
