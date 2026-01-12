"""
Voice Chatbot API - Main Application
A FastAPI-based chatbot with OpenAI integration
"""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import modules
from app.config import CORS_ORIGINS, MODEL_NAME, CACHE_TTL_SECONDS, COST_PER_1K_INPUT_TOKENS, COST_PER_1K_OUTPUT_TOKENS
from app.schemas import ChatMessage, ChatResponse
from app.guardrails import check_guardrails, BLOCKED_WORDS, OFF_TOPIC_PATTERNS
from app.cache import get_cached_response, cache_response, clear_cache, get_cache_stats, RESPONSE_CACHE
from app.matching import match_direct_tool, execute_direct_tool, get_direct_match_stats, DIRECT_TOOL_PATTERNS
from app.tools import TOOL_FUNCTIONS, OPENAI_TOOLS
from app.openai_service import chat_with_openai

# ============================================
# LOGGING CONFIGURATION
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("🚀 Voice Chatbot API starting up...")

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="Voice Chatbot API",
    description="AI-powered travel assistant with voice support",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Voice Chatbot API is running",
        "version": "2.0.0",
        "model": MODEL_NAME
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/api/tools")
async def list_tools():
    """List all available tools"""
    return {
        "tools": [
            {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"]
            }
            for tool in OPENAI_TOOLS
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
            "typical_request_cost": "$0.0001 - $0.0005"
        }
    }

@app.get("/api/cache")
async def cache_statistics():
    """Get cache statistics"""
    stats = get_cache_stats()
    total_requests = stats["stats"]["hits"] + stats["stats"]["misses"]
    hit_rate = (stats["stats"]["hits"] / total_requests * 100) if total_requests > 0 else 0
    
    return {
        "cache_entries": stats["entries"],
        "hits": stats["stats"]["hits"],
        "misses": stats["stats"]["misses"],
        "hit_rate_percent": round(hit_rate, 2),
        "estimated_cost_saved": f"${stats['stats']['saved_cost']:.4f}",
        "ttl_settings": CACHE_TTL_SECONDS
    }

@app.delete("/api/cache")
async def clear_cache_endpoint():
    """Clear the response cache"""
    entries_cleared = clear_cache()
    return {"message": f"Cache cleared. Removed {entries_cleared} entries."}

@app.get("/api/stats")
async def usage_stats():
    """Get comprehensive usage statistics"""
    cache_stats = get_cache_stats()
    direct_stats = get_direct_match_stats()
    
    total_requests = cache_stats["stats"]["hits"] + cache_stats["stats"]["misses"]
    cache_hit_rate = (cache_stats["stats"]["hits"] / total_requests * 100) if total_requests > 0 else 0
    
    return {
        "model": MODEL_NAME,
        "cache": {
            "entries": cache_stats["entries"],
            "hits": cache_stats["stats"]["hits"],
            "misses": cache_stats["stats"]["misses"],
            "hit_rate": f"{cache_hit_rate:.1f}%",
            "cost_saved": f"${cache_stats['stats']['saved_cost']:.4f}"
        },
        "direct_matching": {
            "matches": direct_stats["matches"],
            "api_calls_saved": direct_stats["api_calls_saved"],
            "cost_saved": f"${direct_stats['cost_saved']:.4f}"
        },
        "total_savings": f"${cache_stats['stats']['saved_cost'] + direct_stats['cost_saved']:.4f}",
        "supported_direct_tools": list(DIRECT_TOOL_PATTERNS.keys())
    }

# ============================================
# MAIN CHAT ENDPOINT
# ============================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Process chat message with optimizations:
    1. Guardrails check (block inappropriate content)
    2. Cache check (return cached response)
    3. Direct tool matching (handle common queries FREE)
    4. OpenAI API call (complex queries only)
    """
    try:
        logger.info(f"📨 Received message: {message.message[:100]}")
        
        # ============================================
        # STEP 1: GUARDRAILS CHECK
        # ============================================
        is_allowed, rejection_message = check_guardrails(message.message)
        if not is_allowed:
            logger.info(f"🛡️ Guardrail blocked request - No API cost")
            return ChatResponse(response=rejection_message, cost_usd=0.0)
        
        # ============================================
        # STEP 2: CACHE CHECK
        # ============================================
        cached_response = get_cached_response(message.message)
        if cached_response:
            return ChatResponse(response=cached_response, cost_usd=0.0)
        
        # ============================================
        # STEP 3: DIRECT TOOL MATCHING (FREE!)
        # ============================================
        tool_name, tool_params = match_direct_tool(message.message)
        
        if tool_name:
            response_text = execute_direct_tool(tool_name, tool_params)
            if response_text:
                cache_response(message.message, response_text)
                return ChatResponse(response=response_text, cost_usd=0.0)
        
        # ============================================
        # STEP 4: OPENAI API CALL (PAID)
        # ============================================
        logger.info(f"💵 PAID - No direct match, calling OpenAI API")
        
        response_text, total_cost = chat_with_openai(message.message, message.history)
        
        # Cache the response
        cache_response(message.message, response_text)
        
        return ChatResponse(response=response_text, cost_usd=total_cost)
        
    except Exception as e:
        logger.error(f"❌ Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
