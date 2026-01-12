"""
Vercel Serverless Entry Point - Self-contained version
All code in one file for Vercel compatibility
"""
import os
import json
import hashlib
import re
import logging
from time import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

MODEL_NAME = "gpt-4o-mini"
COST_PER_1K_INPUT_TOKENS = 0.00015
COST_PER_1K_OUTPUT_TOKENS = 0.0006
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CORS_ORIGINS = ["http://localhost:3000", "https://yourdomain.com", "*"]

CACHE_TTL_SECONDS = {
    "time": 60,
    "date": 3600,
    "general": 3600 * 6,
    "static": 3600 * 24,
}

MAX_CACHE_SIZE = 1000

# ============================================
# LOGGING
# ============================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# SCHEMAS
# ============================================

class ChatMessage(BaseModel):
    message: str
    history: List[dict] = []

class ChatResponse(BaseModel):
    response: str
    cost_usd: float = 0.0

# ============================================
# GUARDRAILS
# ============================================

BLOCKED_WORDS = {
    'fuck', 'shit', 'damn', 'ass', 'bitch', 'bastard', 'crap', 'dick', 'pussy',
    'cock', 'cunt', 'whore', 'slut', 'nigger', 'faggot', 'retard',
    'kill', 'murder', 'bomb', 'attack', 'terrorist', 'weapon', 'gun',
    'porn', 'xxx', 'nude', 'naked', 'sex',
}

OFF_TOPIC_PATTERNS = [
    'hack', 'crack', 'pirate', 'illegal', 'drugs', 'weed', 'cocaine',
    'password', 'exploit', 'malware', 'virus',
    'dating', 'girlfriend', 'boyfriend', 'marry me',
    'gambling', 'casino', 'bet', 'lottery',
]

def check_guardrails(message: str) -> Tuple[bool, str]:
    message_lower = message.lower()
    words = set(message_lower.split())
    
    if words.intersection(BLOCKED_WORDS):
        return False, "I'm sorry, but I can't respond to messages with inappropriate language."
    
    for pattern in OFF_TOPIC_PATTERNS:
        if pattern in message_lower:
            return False, "I'm a travel assistant. I can help you with tours, bus fares, and bookings."
    
    if len(message.strip()) < 2:
        return False, "Please provide a complete question."
    
    if len(message) > 1000:
        return False, "Your message is too long. Please keep it under 1000 characters."
    
    return True, ""

# ============================================
# CACHE
# ============================================

RESPONSE_CACHE = {}
cache_stats = {"hits": 0, "misses": 0, "saved_cost": 0.0}

def get_cache_key(query: str) -> str:
    normalized = re.sub(r'\s+', ' ', query.lower().strip())
    normalized = re.sub(r'[^\w\s]', '', normalized)
    return hashlib.md5(normalized.encode()).hexdigest()

def get_cached_response(query: str) -> Optional[str]:
    cache_key = get_cache_key(query)
    if cache_key in RESPONSE_CACHE:
        cached = RESPONSE_CACHE[cache_key]
        if time() - cached["timestamp"] < CACHE_TTL_SECONDS["general"]:
            cached["hits"] += 1
            cache_stats["hits"] += 1
            cache_stats["saved_cost"] += 0.001
            return cached["response"]
        del RESPONSE_CACHE[cache_key]
    cache_stats["misses"] += 1
    return None

def cache_response(query: str, response: str):
    if len(RESPONSE_CACHE) >= MAX_CACHE_SIZE:
        oldest_key = min(RESPONSE_CACHE, key=lambda k: RESPONSE_CACHE[k]["timestamp"])
        del RESPONSE_CACHE[oldest_key]
    RESPONSE_CACHE[get_cache_key(query)] = {"response": response, "timestamp": time(), "hits": 0}

# ============================================
# TOOLS
# ============================================

def get_current_time() -> str:
    return f"The current time is {datetime.now().strftime('%I:%M %p')}"

def get_current_date() -> str:
    return f"Today's date is {datetime.now().strftime('%B %d, %Y')}"

def get_recent_tours() -> str:
    tours = [
        {"name": "Kashmir Valley Tour", "duration": "7 days", "departure": "Feb 15, 2026", "price": "₹25,000"},
        {"name": "Goa Beach Paradise", "duration": "5 days", "departure": "Feb 20, 2026", "price": "₹15,000"},
        {"name": "Rajasthan Heritage Tour", "duration": "10 days", "departure": "Mar 1, 2026", "price": "₹35,000"},
        {"name": "Kerala Backwaters", "duration": "6 days", "departure": "Mar 10, 2026", "price": "₹22,000"},
        {"name": "Himalayan Adventure", "duration": "8 days", "departure": "Mar 25, 2026", "price": "₹30,000"}
    ]
    result = "🌍 Available Tours:\n\n"
    for i, tour in enumerate(tours, 1):
        result += f"{i}. {tour['name']}\n   Duration: {tour['duration']} | Departure: {tour['departure']} | Price: {tour['price']}\n\n"
    return result.strip()

def get_bus_fares() -> str:
    bus_fares = [
        {"type": "Ordinary Bus", "fare": "₹50-80", "features": "Basic seating, frequent stops"},
        {"type": "Express Bus", "fare": "₹100-150", "features": "Limited stops, faster travel"},
        {"type": "Deluxe Bus", "fare": "₹200-300", "features": "Comfortable seats, AC available"},
        {"type": "Super Deluxe", "fare": "₹350-450", "features": "Reclining seats, AC, charging"},
        {"type": "Volvo AC Bus", "fare": "₹500-700", "features": "Premium AC, 2+2 seating"},
        {"type": "Sleeper Bus", "fare": "₹600-900", "features": "Sleeper berths, AC"},
        {"type": "Multi-Axle Volvo", "fare": "₹800-1200", "features": "Luxury, smooth ride"}
    ]
    result = "🚌 Bus Fares:\n\n"
    for bus in bus_fares:
        result += f"• {bus['type']}: {bus['fare']}\n  {bus['features']}\n\n"
    return result.strip()

def get_contact_info() -> str:
    return "📞 Contact: +91 98765 43210\n📧 Email: info@viptravels.com\n📍 Office: 123 Travel Street, New Delhi"

def get_booking_info() -> str:
    return "📋 How to Book:\n1. Online: www.viptravels.com\n2. Phone: +91 98765 43210\n3. WhatsApp: +91 98765 43210"

TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_current_date": get_current_date,
    "get_recent_tours": get_recent_tours,
    "get_bus_fares": get_bus_fares,
    "get_contact_info": get_contact_info,
    "get_booking_info": get_booking_info,
}

# Direct matching patterns
DIRECT_PATTERNS = {
    "get_current_time": ["time", "clock", "what time"],
    "get_current_date": ["date", "today", "what day"],
    "get_recent_tours": ["tour", "tours", "trip", "vacation", "holiday", "package"],
    "get_bus_fares": ["bus", "fare", "fares", "volvo", "sleeper", "deluxe"],
    "get_contact_info": ["contact", "phone", "call", "email", "address"],
    "get_booking_info": ["book", "booking", "reserve", "payment", "how to book"],
}

def match_direct_tool(query: str) -> Optional[str]:
    query_lower = query.lower()
    for tool_name, keywords in DIRECT_PATTERNS.items():
        if any(kw in query_lower for kw in keywords):
            return tool_name
    return None

direct_match_stats = {"matches": 0, "cost_saved": 0.0}

# ============================================
# OPENAI
# ============================================

_client = None

def get_openai_client():
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

OPENAI_TOOLS = [
    {"type": "function", "function": {"name": "get_current_time", "description": "Get current time", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_current_date", "description": "Get today's date", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_recent_tours", "description": "Get available tours", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_bus_fares", "description": "Get bus fares", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_contact_info", "description": "Get contact info", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_booking_info", "description": "Get booking info", "parameters": {"type": "object", "properties": {}, "required": []}}},
]

def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return round((prompt_tokens / 1000) * COST_PER_1K_INPUT_TOKENS + (completion_tokens / 1000) * COST_PER_1K_OUTPUT_TOKENS, 6)

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(title="Voice Chatbot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Voice Chatbot API is running", "version": "2.0.0", "model": MODEL_NAME}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/tools")
async def list_tools():
    return {"tools": [{"name": t["function"]["name"], "description": t["function"]["description"]} for t in OPENAI_TOOLS]}

@app.get("/api/stats")
async def usage_stats():
    return {
        "model": MODEL_NAME,
        "cache": {"entries": len(RESPONSE_CACHE), "hits": cache_stats["hits"], "saved": f"${cache_stats['saved_cost']:.4f}"},
        "direct_matching": {"matches": direct_match_stats["matches"], "saved": f"${direct_match_stats['cost_saved']:.4f}"}
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        # Guardrails
        is_allowed, rejection = check_guardrails(message.message)
        if not is_allowed:
            return ChatResponse(response=rejection, cost_usd=0.0)
        
        # Cache
        cached = get_cached_response(message.message)
        if cached:
            return ChatResponse(response=cached, cost_usd=0.0)
        
        # Direct match
        tool_name = match_direct_tool(message.message)
        if tool_name and tool_name in TOOL_FUNCTIONS:
            direct_match_stats["matches"] += 1
            direct_match_stats["cost_saved"] += 0.001
            response_text = TOOL_FUNCTIONS[tool_name]()
            cache_response(message.message, response_text)
            return ChatResponse(response=response_text, cost_usd=0.0)
        
        # OpenAI
        client = get_openai_client()
        messages = [
            {"role": "system", "content": "You are a helpful travel assistant for VIP Travels."}
        ] + message.history + [{"role": "user", "content": message.message}]
        
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=OPENAI_TOOLS, tool_choice="auto", max_tokens=300
        )
        
        total_cost = calculate_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
        response_message = response.choices[0].message
        
        if response_message.tool_calls:
            tool_results = []
            for tc in response_message.tool_calls:
                fn = tc.function.name
                result = TOOL_FUNCTIONS.get(fn, lambda: "Unknown")()
                tool_results.append({"tool_call_id": tc.id, "role": "tool", "content": result})
            
            messages.append(response_message)
            messages.extend(tool_results)
            
            final = client.chat.completions.create(model=MODEL_NAME, messages=messages, max_tokens=300)
            total_cost += calculate_cost(final.usage.prompt_tokens, final.usage.completion_tokens)
            response_text = final.choices[0].message.content
        else:
            response_text = response_message.content
        
        cache_response(message.message, response_text)
        return ChatResponse(response=response_text, cost_usd=total_cost)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Export for Vercel
handler = app
