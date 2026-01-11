from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

logger.info("🚀 Voice Chatbot API starting up...")

app = FastAPI(title="Voice Chatbot API")

# Initialize OpenAI client lazily
_client = None

def get_openai_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500, 
                detail="OPENAI_API_KEY not set. Please add it to your .env file"
            )
        _client = OpenAI(api_key=api_key)
    return _client

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str
    history: List[dict] = []

class ChatResponse(BaseModel):
    response: str

# Tools
def get_current_time() -> str:
    return f"The current time is {datetime.now().strftime('%I:%M %p')}"

def get_current_date() -> str:
    return f"Today's date is {datetime.now().strftime('%B %d, %Y')}"

def calculate(expression: str) -> str:
    try:
        # Safe evaluation
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is {result}"
    except:
        return "I couldn't calculate that expression"

def get_recent_tours() -> str:
    """Get list of recent tours"""
    tours = [
        "Kashmir Valley Tour - 7 days, departing Feb 15, 2026",
        "Goa Beach Paradise - 5 days, departing Feb 20, 2026",
        "Rajasthan Heritage Tour - 10 days, departing Mar 1, 2026",
        "Kerala Backwaters - 6 days, departing Mar 10, 2026",
        "Himalayan Adventure - 8 days, departing Mar 25, 2026"
    ]
    return "Here are our recent tours:\n" + "\n".join([f"{i+1}. {tour}" for i, tour in enumerate(tours)])

TOOLS = {
    'time': get_current_time,
    'date': get_current_date,
    'calculate': calculate,
    'tours': get_recent_tours
}

def detect_tool_intent(text: str):
    """Detect if message requires a tool"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['time', 'what time']):
        return 'time', None
    elif any(word in text_lower for word in ['date', 'today', 'what day']):
        return 'date', None
    elif any(word in text_lower for word in ['calculate', 'compute', 'what is']) and \
         any(op in text for op in ['+', '-', '*', '/', 'plus', 'minus']):
        return 'calculate', text_lower
    elif any(word in text_lower for word in ['tour', 'tours', 'trip', 'trips', 'travel', 'package']):
        return 'tours', None
    
    return None, None

@app.get("/")
async def root():
    return {"message": "Voice Chatbot API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/check-config")
async def check_config():
    """Check if API key is configured"""
    api_key = os.getenv("OPENAI_API_KEY")
    return {
        "api_key_configured": bool(api_key),
        "api_key_length": len(api_key) if api_key else 0,
        "api_key_prefix": api_key[:7] if api_key else "not set"
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Process chat message and return response"""
    try:
        logger.info(f"📨 Received message: {message.message[:100]}")
        
        # Check for tool intent
        tool_name, tool_params = detect_tool_intent(message.message)
        
        if tool_name and tool_name in TOOLS:
            logger.info(f"✅ FREE - Using tool: {tool_name}")
            if tool_params:
                response_text = TOOLS[tool_name](tool_params)
            else:
                response_text = TOOLS[tool_name]()
            logger.info(f"✅ Tool response: {response_text[:100]}")
            return ChatResponse(response=response_text)
        
        # Use OpenAI for general conversation
        logger.info(f"💵 PAID - Calling OpenAI API for general conversation")
        messages = [
            {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise and conversational."}
        ] + message.history + [{"role": "user", "content": message.message}]
        
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content
        logger.info(f"💵 OpenAI response: {response_text[:100]}")
        logger.info(f"💰 Tokens used: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})")
        
        return ChatResponse(response=response_text)
    except Exception as e:
        logger.error(f"❌ Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
