"""
Google Gemini client and chat service
"""
import logging
import google.generativeai as genai
from typing import Optional, List, Dict, Any
from fastapi import HTTPException
from google.generativeai.types import FunctionDeclaration, Tool

from app.config import GOOGLE_API_KEY, GEMINI_MODEL_NAME
from app.tools import TOOL_FUNCTIONS

logger = logging.getLogger(__name__)

# ============================================
# GEMINI CLIENT (LAZY INITIALIZATION)
# ============================================

_model = None

def get_gemini_model():
    """Get or create Gemini model instance"""
    global _model
    if _model is None:
        if not GOOGLE_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GOOGLE_API_KEY not set. Please add it to your .env file"
            )
        
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Define tools for Gemini
        # Gemini uses a different format for tools than OpenAI
        # We need to wrap our Python functions directly or define FunctionDeclarations
        
        # Option 1: Pass the actual functions (easiest for Python SDK)
        tools = list(TOOL_FUNCTIONS.values())
        
        _model = genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            tools=tools
        )
        
    return _model

# ============================================
# CHAT SERVICE
# ============================================

def chat_with_gemini(message: str, history: list) -> tuple[str, float]:
    """
    Process message using Google Gemini with function calling.
    Returns: (response_text, estimated_cost)
    """
    try:
        model = get_gemini_model()
        
        # Convert history to Gemini format if needed
        # Gemini handles history via ChatSession usually, or we can pass context
        # For simplicity in this stateless API, we'll start a new chat with history if feasible,
        # or just append history to the prompt context.
        
        # Simplified approach: Start a chat session
        chat_session = model.start_chat(enable_automatic_function_calling=True)
        
        # Add system context (Gemini Pro doesn't rely heavily on system prompts in start_chat)
        # We can prepend it to the first message or rely on the tool definitions context.
        system_instruction = """You are a helpful travel assistant for VIP Travels. 
        You help users with tour bookings, bus fares, and travel information.
        Use the available tools to provide accurate information.
        Keep responses concise and friendly.
        If asked about topics not related to travel, politely redirect to travel topics."""
        
        # Send message
        # We prepend system instruction to the user message for context
        full_message = f"{system_instruction}\n\nUser Query: {message}"
        
        response = chat_session.send_message(full_message)
        
        response_text = response.text
        
        # Cost estimation (Gemini Pro is free-ish for now within limits, or pay-as-you-go)
        # 1K input chars ~= $0.000125, 1K output chars ~= $0.000375 (pricing varies)
        # For now, we'll return 0.0 or a rough estimate
        estimated_cost = 0.0001
        
        logger.info(f"💬 Gemini Response: {response_text[:100]}")
        
        return response_text, estimated_cost

    except Exception as e:
        logger.error(f"❌ Gemini Error: {str(e)}")
        # Fallback to a generic error message or re-raise
        raise HTTPException(status_code=500, detail=f"Gemini Error: {str(e)}")
