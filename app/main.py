from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import json
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
    cost_usd: float = 0.0  # Optional cost tracking

# ============================================
# GUARDRAILS - Block inappropriate content
# ============================================

# Foul/inappropriate words list (add more as needed)
BLOCKED_WORDS = {
    # Profanity
    'fuck', 'shit', 'damn', 'ass', 'bitch', 'bastard', 'crap', 'dick', 'pussy',
    'cock', 'cunt', 'whore', 'slut', 'nigger', 'faggot', 'retard',
    # Violence
    'kill', 'murder', 'bomb', 'attack', 'terrorist', 'weapon', 'gun',
    # Other inappropriate
    'porn', 'xxx', 'nude', 'naked', 'sex',
}

# Off-topic categories (not related to travel/tours/buses)
OFF_TOPIC_PATTERNS = [
    'hack', 'crack', 'pirate', 'illegal', 'drugs', 'weed', 'cocaine',
    'password', 'exploit', 'malware', 'virus',
    'dating', 'girlfriend', 'boyfriend', 'marry me',
    'gambling', 'casino', 'bet', 'lottery',
]

def check_guardrails(message: str) -> tuple[bool, str]:
    """
    Check if message passes guardrails.
    Returns: (is_allowed, rejection_message)
    """
    message_lower = message.lower()
    words = set(message_lower.split())
    
    # Check for blocked words
    found_blocked = words.intersection(BLOCKED_WORDS)
    if found_blocked:
        logger.warning(f"🚫 BLOCKED - Foul language detected: {found_blocked}")
        return False, "I'm sorry, but I can't respond to messages with inappropriate language. Please rephrase your question politely."
    
    # Check for off-topic patterns
    for pattern in OFF_TOPIC_PATTERNS:
        if pattern in message_lower:
            logger.warning(f"🚫 BLOCKED - Off-topic content detected: {pattern}")
            return False, "I'm a travel assistant for VIP Travels. I can help you with tours, bus fares, bookings, and travel information. How can I assist you with your travel needs?"
    
    # Check for very short messages (likely spam)
    if len(message.strip()) < 2:
        logger.warning(f"🚫 BLOCKED - Message too short")
        return False, "Please provide a complete question. How can I help you with travel information?"
    
    # Check for very long messages (potential abuse)
    if len(message) > 1000:
        logger.warning(f"🚫 BLOCKED - Message too long: {len(message)} chars")
        return False, "Your message is too long. Please keep your question concise (under 1000 characters)."
    
    return True, ""

# ============================================
# MODEL CONFIGURATION - Using cheapest model
# ============================================

# gpt-4o-mini is the CHEAPEST OpenAI model that supports function calling
# It's 3x cheaper than gpt-3.5-turbo!
MODEL_NAME = "gpt-4o-mini"

# ============================================
# COST TRACKING
# ============================================

# gpt-4o-mini pricing (per 1K tokens) - CHEAPEST!
COST_PER_1K_INPUT_TOKENS = 0.00015   # $0.15 per 1M = $0.00015 per 1K
COST_PER_1K_OUTPUT_TOKENS = 0.0006   # $0.60 per 1M = $0.0006 per 1K

def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD for API call"""
    input_cost = (prompt_tokens / 1000) * COST_PER_1K_INPUT_TOKENS
    output_cost = (completion_tokens / 1000) * COST_PER_1K_OUTPUT_TOKENS
    total_cost = input_cost + output_cost
    return round(total_cost, 6)

# ============================================
# CACHING SYSTEM - Save costs on repeated queries
# ============================================

import hashlib
import re
from typing import Optional
from time import time

# Cache storage: {normalized_query: {"response": str, "timestamp": float, "hits": int}}
RESPONSE_CACHE = {}

# Cache settings
CACHE_TTL_SECONDS = {
    "time": 60,          # Time queries expire after 1 minute
    "date": 3600,        # Date queries expire after 1 hour
    "general": 3600 * 6, # General responses expire after 6 hours
    "static": 3600 * 24, # Static content (tours, fares) expire after 24 hours
}

# Maximum cache size
MAX_CACHE_SIZE = 1000

# Cache statistics
cache_stats = {
    "hits": 0,
    "misses": 0,
    "saved_cost": 0.0
}

def normalize_query(query: str) -> str:
    """Normalize query for cache matching"""
    # Convert to lowercase
    normalized = query.lower().strip()
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    # Remove punctuation
    normalized = re.sub(r'[^\w\s]', '', normalized)
    # Sort words for order-independent matching (optional)
    # words = sorted(normalized.split())
    # normalized = ' '.join(words)
    return normalized

def get_cache_key(query: str) -> str:
    """Generate cache key from normalized query"""
    normalized = normalize_query(query)
    return hashlib.md5(normalized.encode()).hexdigest()

def get_cache_ttl(query: str) -> int:
    """Determine cache TTL based on query type"""
    query_lower = query.lower()
    
    # Time-sensitive queries - short TTL
    if any(word in query_lower for word in ['time', 'clock', 'now']):
        return CACHE_TTL_SECONDS["time"]
    
    # Date queries - medium TTL
    if any(word in query_lower for word in ['date', 'today', 'day']):
        return CACHE_TTL_SECONDS["date"]
    
    # Static content - long TTL
    if any(word in query_lower for word in ['tour', 'bus', 'fare', 'price', 'contact', 'booking']):
        return CACHE_TTL_SECONDS["static"]
    
    # General queries
    return CACHE_TTL_SECONDS["general"]

def get_cached_response(query: str) -> Optional[str]:
    """Get response from cache if available and not expired"""
    cache_key = get_cache_key(query)
    
    if cache_key in RESPONSE_CACHE:
        cached = RESPONSE_CACHE[cache_key]
        ttl = get_cache_ttl(query)
        
        # Check if cache is still valid
        if time() - cached["timestamp"] < ttl:
            cached["hits"] += 1
            cache_stats["hits"] += 1
            cache_stats["saved_cost"] += 0.001  # Approximate cost saved per hit
            logger.info(f"📦 CACHE HIT - Query matched (hits: {cached['hits']}, saved: ${cache_stats['saved_cost']:.4f})")
            return cached["response"]
        else:
            # Cache expired, remove it
            del RESPONSE_CACHE[cache_key]
            logger.info(f"📦 CACHE EXPIRED - Removing stale entry")
    
    cache_stats["misses"] += 1
    return None

def cache_response(query: str, response: str):
    """Store response in cache"""
    # Limit cache size
    if len(RESPONSE_CACHE) >= MAX_CACHE_SIZE:
        # Remove oldest entries
        oldest_key = min(RESPONSE_CACHE, key=lambda k: RESPONSE_CACHE[k]["timestamp"])
        del RESPONSE_CACHE[oldest_key]
        logger.info(f"📦 CACHE CLEANUP - Removed oldest entry")
    
    cache_key = get_cache_key(query)
    RESPONSE_CACHE[cache_key] = {
        "response": response,
        "timestamp": time(),
        "hits": 0,
        "original_query": query[:100]  # Store for debugging
    }
    logger.info(f"📦 CACHE STORED - Query cached (total entries: {len(RESPONSE_CACHE)})")

# ============================================
# DIRECT TOOL MATCHING - FREE! No OpenAI call
# ============================================
# Common queries are matched by keywords and handled directly
# This saves money by avoiding OpenAI API calls

# Keyword patterns for direct tool matching
DIRECT_TOOL_PATTERNS = {
    "get_current_time": {
        "keywords": ["time", "clock", "what time", "current time", "time now"],
        "requires_params": False
    },
    "get_current_date": {
        "keywords": ["date", "today", "what day", "today's date", "current date", "which day"],
        "requires_params": False
    },
    "get_recent_tours": {
        "keywords": ["tour", "tours", "trip", "trips", "vacation", "holiday", "package", "travel package", 
                     "upcoming tour", "available tour", "show tour", "list tour", "what tour"],
        "requires_params": False
    },
    "get_bus_fares": {
        "keywords": ["bus", "fare", "fares", "bus fare", "bus price", "bus cost", "bus ticket", 
                     "volvo", "sleeper", "deluxe bus", "ordinary bus", "express bus", "bus rate"],
        "requires_params": False
    },
    "get_contact_info": {
        "keywords": ["contact", "phone", "call", "email", "address", "office", "reach you", 
                     "phone number", "contact number", "how to contact", "where are you"],
        "requires_params": False
    },
    "get_booking_info": {
        "keywords": ["book", "booking", "reserve", "reservation", "how to book", "payment", 
                     "pay", "book tour", "book bus", "make booking", "book ticket"],
        "requires_params": False
    },
    "calculate": {
        "keywords": ["calculate", "compute", "math", "add", "subtract", "multiply", "divide"],
        "operators": ["+", "-", "*", "/", "plus", "minus", "times", "divided"],
        "requires_params": True
    }
}

def match_direct_tool(query: str) -> tuple:
    """
    Match query to a tool using keywords (FREE - no OpenAI call).
    Returns: (tool_name, params) or (None, None)
    """
    query_lower = query.lower().strip()
    
    for tool_name, config in DIRECT_TOOL_PATTERNS.items():
        keywords = config.get("keywords", [])
        
        # Check if any keyword matches
        if any(kw in query_lower for kw in keywords):
            # Special handling for calculator - needs operators
            if tool_name == "calculate":
                operators = config.get("operators", [])
                if any(op in query_lower for op in operators):
                    # Extract expression from query
                    # Try to find numbers and operators
                    import re
                    numbers = re.findall(r'\d+\.?\d*', query)
                    if len(numbers) >= 2:
                        # Try to build expression
                        for op in ['+', '-', '*', '/']:
                            if op in query:
                                return tool_name, {"expression": f"{numbers[0]} {op} {numbers[1]}"}
                        # Check word operators
                        op_map = {"plus": "+", "minus": "-", "times": "*", "divided": "/", "multiply": "*"}
                        for word, op in op_map.items():
                            if word in query_lower:
                                return tool_name, {"expression": f"{numbers[0]} {op} {numbers[1]}"}
            else:
                # Non-parameterized tools
                return tool_name, None
    
    return None, None

# Stats for direct matching
direct_match_stats = {
    "matches": 0,
    "api_calls_saved": 0,
    "cost_saved": 0.0
}

# ============================================
# TOOL FUNCTIONS - Add your custom tools here
# ============================================

def get_current_time() -> str:
    """Returns the current time"""
    return f"The current time is {datetime.now().strftime('%I:%M %p')}"

def get_current_date() -> str:
    """Returns today's date"""
    return f"Today's date is {datetime.now().strftime('%B %d, %Y')}"

def calculate(expression: str) -> str:
    """Calculates a mathematical expression"""
    try:
        # Safe evaluation - only allow basic math
        allowed_chars = set('0123456789+-*/.(). ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression, {"__builtins__": {}}, {})
            return f"The result of {expression} is {result}"
        else:
            return "I can only calculate basic math expressions with numbers and +, -, *, /"
    except Exception as e:
        return f"I couldn't calculate that expression: {str(e)}"

def get_recent_tours() -> str:
    """Returns a list of available travel tours with dates and durations"""
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
    """Returns bus fares and ticket prices for different bus types"""
    bus_fares = [
        {"type": "Ordinary Bus", "fare": "₹50-80", "features": "Basic seating, frequent stops"},
        {"type": "Express Bus", "fare": "₹100-150", "features": "Limited stops, faster travel"},
        {"type": "Deluxe Bus", "fare": "₹200-300", "features": "Comfortable seats, AC available"},
        {"type": "Super Deluxe Bus", "fare": "₹350-450", "features": "Reclining seats, AC, charging points"},
        {"type": "Volvo AC Bus", "fare": "₹500-700", "features": "Premium AC, 2+2 seating, entertainment"},
        {"type": "Sleeper Bus", "fare": "₹600-900", "features": "Sleeper berths, AC, overnight travel"},
        {"type": "Multi-Axle Volvo", "fare": "₹800-1200", "features": "Luxury, smooth ride, premium amenities"}
    ]
    
    result = "🚌 Bus Fares for Different Types:\n\n"
    for bus in bus_fares:
        result += f"• {bus['type']}: {bus['fare']}\n  Features: {bus['features']}\n\n"
    
    return result.strip()

def get_contact_info() -> str:
    """Returns contact information, phone numbers, email, and office address"""
    return """📞 Contact Information:

• Phone: +91 98765 43210
• WhatsApp: +91 98765 43210
• Email: info@viptravels.com
• Office: 123 Travel Street, New Delhi, India 110001

Office Hours: Mon-Sat, 9:00 AM - 6:00 PM"""

def get_booking_info() -> str:
    """Returns information about how to book tours or buses"""
    return """📋 How to Book:

1. Online Booking: Visit our website www.viptravels.com
2. Phone Booking: Call +91 98765 43210
3. WhatsApp: Send your requirements to +91 98765 43210
4. Walk-in: Visit our office at 123 Travel Street, New Delhi

Payment Options:
• UPI, Credit/Debit Cards, Net Banking
• EMI available for tours above ₹10,000
• 10% advance required for tour bookings"""

# ============================================
# TOOL DEFINITIONS FOR OPENAI FUNCTION CALLING
# ============================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time. Use this when user asks about time, what time is it, or current time.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get today's date. Use this when user asks about today's date, what day it is, or current date.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate a mathematical expression. Use this for any math questions, calculations, arithmetic, or when user wants to add, subtract, multiply, or divide numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to calculate, e.g., '25 * 4' or '100 + 50'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_tours",
            "description": "Get list of available travel tours, vacation packages, trips, and holidays. Use this when user asks about tours, trips, travel packages, vacation options, or holiday destinations.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_bus_fares",
            "description": "Get bus ticket prices and fares for different bus types like ordinary, express, deluxe, Volvo, sleeper buses. Use this when user asks about bus fares, ticket prices, bus costs, or bus types.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_contact_info",
            "description": "Get contact information including phone number, email, WhatsApp, and office address. Use this when user wants to contact, call, email, or visit the office.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_booking_info",
            "description": "Get information about how to book tours or buses, payment options, and booking process. Use this when user asks about booking, reservations, how to book, or payment methods.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

# Map function names to actual functions
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_current_date": get_current_date,
    "calculate": calculate,
    "get_recent_tours": get_recent_tours,
    "get_bus_fares": get_bus_fares,
    "get_contact_info": get_contact_info,
    "get_booking_info": get_booking_info
}

# ============================================
# API ENDPOINTS
# ============================================

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

@app.get("/api/tools")
async def list_tools():
    """List all available tools"""
    return {
        "tools": [
            {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"]
            }
            for tool in TOOLS
        ]
    }

@app.get("/api/guardrails")
async def guardrails_info():
    """Get information about content guardrails"""
    return {
        "blocked_word_count": len(BLOCKED_WORDS),
        "off_topic_pattern_count": len(OFF_TOPIC_PATTERNS),
        "min_message_length": 2,
        "max_message_length": 1000,
        "pricing": {
            "model": MODEL_NAME,
            "input_per_1k_tokens": f"${COST_PER_1K_INPUT_TOKENS}",
            "output_per_1k_tokens": f"${COST_PER_1K_OUTPUT_TOKENS}",
            "typical_request_cost": "$0.0005 - $0.002"
        }
    }

@app.get("/api/cache")
async def cache_statistics():
    """Get cache statistics and savings"""
    hit_rate = (cache_stats["hits"] / (cache_stats["hits"] + cache_stats["misses"]) * 100) if (cache_stats["hits"] + cache_stats["misses"]) > 0 else 0
    
    return {
        "cache_entries": len(RESPONSE_CACHE),
        "max_cache_size": MAX_CACHE_SIZE,
        "hits": cache_stats["hits"],
        "misses": cache_stats["misses"],
        "hit_rate_percent": round(hit_rate, 2),
        "estimated_cost_saved": f"${cache_stats['saved_cost']:.4f}",
        "ttl_settings": CACHE_TTL_SECONDS
    }

@app.delete("/api/cache")
async def clear_cache():
    """Clear the response cache"""
    global RESPONSE_CACHE, cache_stats
    entries_cleared = len(RESPONSE_CACHE)
    RESPONSE_CACHE = {}
    cache_stats = {"hits": 0, "misses": 0, "saved_cost": 0.0}
    logger.info(f"📦 CACHE CLEARED - Removed {entries_cleared} entries")
    return {"message": f"Cache cleared. Removed {entries_cleared} entries."}

@app.get("/api/stats")
async def usage_stats():
    """Get comprehensive usage statistics"""
    total_requests = cache_stats["hits"] + cache_stats["misses"]
    cache_hit_rate = (cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
    
    return {
        "model": MODEL_NAME,
        "cache": {
            "entries": len(RESPONSE_CACHE),
            "hits": cache_stats["hits"],
            "misses": cache_stats["misses"],
            "hit_rate": f"{cache_hit_rate:.1f}%",
            "cost_saved": f"${cache_stats['saved_cost']:.4f}"
        },
        "direct_matching": {
            "matches": direct_match_stats["matches"],
            "api_calls_saved": direct_match_stats["api_calls_saved"],
            "cost_saved": f"${direct_match_stats['cost_saved']:.4f}"
        },
        "total_savings": f"${cache_stats['saved_cost'] + direct_match_stats['cost_saved']:.4f}",
        "supported_direct_tools": list(DIRECT_TOOL_PATTERNS.keys())
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Process chat message using OpenAI function calling with guardrails and caching"""
    try:
        logger.info(f"📨 Received message: {message.message[:100]}")
        
        # ============================================
        # GUARDRAILS CHECK - Block before API call
        # ============================================
        is_allowed, rejection_message = check_guardrails(message.message)
        if not is_allowed:
            logger.info(f"🛡️ Guardrail blocked request - No API cost incurred")
            return ChatResponse(response=rejection_message, cost_usd=0.0)
        
        # ============================================
        # CACHE CHECK - Return cached response if available
        # ============================================
        cached_response = get_cached_response(message.message)
        if cached_response:
            return ChatResponse(response=cached_response, cost_usd=0.0)
        
        # ============================================
        # DIRECT TOOL MATCHING - FREE! No OpenAI call
        # ============================================
        # Handle common queries directly without calling OpenAI
        tool_name, tool_params = match_direct_tool(message.message)
        
        if tool_name and tool_name in TOOL_FUNCTIONS:
            logger.info(f"🆓 FREE - Direct tool match: {tool_name}")
            direct_match_stats["matches"] += 1
            direct_match_stats["api_calls_saved"] += 1
            direct_match_stats["cost_saved"] += 0.001  # Approximate savings
            
            # Execute the tool directly
            func = TOOL_FUNCTIONS[tool_name]
            if tool_params:
                response_text = func(**tool_params)
            else:
                response_text = func()
            
            logger.info(f"🆓 Direct tool result: {response_text[:100]}...")
            logger.info(f"💚 API calls saved: {direct_match_stats['api_calls_saved']}, Total saved: ${direct_match_stats['cost_saved']:.4f}")
            
            # Cache the response
            cache_response(message.message, response_text)
            
            return ChatResponse(response=response_text, cost_usd=0.0)
        
        # ============================================
        # OPENAI API CALL - Only for complex queries
        # ============================================
        logger.info(f"💵 PAID - No direct match, calling OpenAI API")
        
        client = get_openai_client()
        total_cost = 0.0
        
        # Build messages with history
        messages = [
            {
                "role": "system", 
                "content": """You are a helpful travel assistant for VIP Travels. 
                You help users with tour bookings, bus fares, and travel information.
                Use the available tools to provide accurate information.
                Keep responses concise and friendly.
                If asked about topics not related to travel, politely redirect to travel topics."""
            }
        ] + message.history + [{"role": "user", "content": message.message}]
        
        # Call OpenAI with function calling
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=300,
            temperature=0.7
        )
        
        # Calculate cost for first call
        cost1 = calculate_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
        total_cost += cost1
        
        response_message = response.choices[0].message
        
        # Check if model wants to call a function
        if response_message.tool_calls:
            logger.info(f"🔧 AI wants to call tools: {[tc.function.name for tc in response_message.tool_calls]}")
            
            # Execute all tool calls
            tool_results = []
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                
                logger.info(f"✅ Executing tool: {function_name} with args: {function_args}")
                
                # Get the function and execute it
                if function_name in TOOL_FUNCTIONS:
                    func = TOOL_FUNCTIONS[function_name]
                    if function_args:
                        result = func(**function_args)
                    else:
                        result = func()
                else:
                    result = f"Unknown function: {function_name}"
                
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": result
                })
                
                logger.info(f"✅ Tool result: {result[:100]}...")
            
            # Send tool results back to get final response
            messages.append(response_message)
            messages.extend(tool_results)
            
            final_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            
            # Calculate cost for second call
            cost2 = calculate_cost(final_response.usage.prompt_tokens, final_response.usage.completion_tokens)
            total_cost += cost2
            
            response_text = final_response.choices[0].message.content
            
            # Log cost breakdown
            logger.info(f"💬 Final response: {response_text[:100]}")
            logger.info(f"💰 Cost breakdown:")
            logger.info(f"   📊 Call 1: {response.usage.prompt_tokens} input + {response.usage.completion_tokens} output = ${cost1:.6f}")
            logger.info(f"   📊 Call 2: {final_response.usage.prompt_tokens} input + {final_response.usage.completion_tokens} output = ${cost2:.6f}")
            logger.info(f"   💵 Total cost: ${total_cost:.6f}")
            
            # Cache the response for future queries
            cache_response(message.message, response_text)
            
            return ChatResponse(response=response_text, cost_usd=total_cost)
        
        # No function call, return direct response
        response_text = response_message.content
        
        logger.info(f"💬 Direct response: {response_text[:100]}")
        logger.info(f"💰 Cost: {response.usage.prompt_tokens} input + {response.usage.completion_tokens} output = ${total_cost:.6f}")
        
        # Cache the response for future queries
        cache_response(message.message, response_text)
        
        return ChatResponse(response=response_text, cost_usd=total_cost)
        
    except Exception as e:
        logger.error(f"❌ Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
