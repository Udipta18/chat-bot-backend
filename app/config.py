"""
Configuration settings for the Voice Chatbot API
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# MODEL CONFIGURATION
# ============================================

# gpt-4o-mini is the CHEAPEST OpenAI model that supports function calling
MODEL_NAME = "gpt-4o-mini"

# ============================================
# COST TRACKING
# ============================================

# gpt-4o-mini pricing (per 1K tokens)
COST_PER_1K_INPUT_TOKENS = 0.00015   # $0.15 per 1M = $0.00015 per 1K
COST_PER_1K_OUTPUT_TOKENS = 0.0006   # $0.60 per 1M = $0.0006 per 1K

def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD for API call"""
    input_cost = (prompt_tokens / 1000) * COST_PER_1K_INPUT_TOKENS
    output_cost = (completion_tokens / 1000) * COST_PER_1K_OUTPUT_TOKENS
    total_cost = input_cost + output_cost
    return round(total_cost, 6)

# ============================================
# API SETTINGS
# ============================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# CORS origins
CORS_ORIGINS = [
    "http://localhost:3000",
    "https://yourdomain.com",
    "*"
]

# ============================================
# CACHE SETTINGS
# ============================================

CACHE_TTL_SECONDS = {
    "time": 60,          # Time queries expire after 1 minute
    "date": 3600,        # Date queries expire after 1 hour
    "general": 3600 * 6, # General responses expire after 6 hours
    "static": 3600 * 24, # Static content (tours, fares) expire after 24 hours
}

MAX_CACHE_SIZE = 1000
