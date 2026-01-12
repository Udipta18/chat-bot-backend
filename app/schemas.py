"""
Pydantic models for request/response schemas
"""
from pydantic import BaseModel
from typing import List

class ChatMessage(BaseModel):
    """Request model for chat endpoint"""
    message: str
    history: List[dict] = []

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    cost_usd: float = 0.0
