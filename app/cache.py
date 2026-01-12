"""
Caching system for responses
Saves costs by caching repeated queries
"""
import hashlib
import re
import logging
from time import time
from typing import Optional

from app.config import CACHE_TTL_SECONDS, MAX_CACHE_SIZE

logger = logging.getLogger(__name__)

# ============================================
# CACHE STORAGE
# ============================================

# Cache storage: {normalized_query: {"response": str, "timestamp": float, "hits": int}}
RESPONSE_CACHE = {}

# Cache statistics
cache_stats = {
    "hits": 0,
    "misses": 0,
    "saved_cost": 0.0
}

# ============================================
# CACHE FUNCTIONS
# ============================================

def normalize_query(query: str) -> str:
    """Normalize query for cache matching"""
    normalized = query.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    return normalized

def get_cache_key(query: str) -> str:
    """Generate cache key from normalized query"""
    normalized = normalize_query(query)
    return hashlib.md5(normalized.encode()).hexdigest()

def get_cache_ttl(query: str) -> int:
    """Determine cache TTL based on query type"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['time', 'clock', 'now']):
        return CACHE_TTL_SECONDS["time"]
    
    if any(word in query_lower for word in ['date', 'today', 'day']):
        return CACHE_TTL_SECONDS["date"]
    
    if any(word in query_lower for word in ['tour', 'bus', 'fare', 'price', 'contact', 'booking']):
        return CACHE_TTL_SECONDS["static"]
    
    return CACHE_TTL_SECONDS["general"]

def get_cached_response(query: str) -> Optional[str]:
    """Get response from cache if available and not expired"""
    cache_key = get_cache_key(query)
    
    if cache_key in RESPONSE_CACHE:
        cached = RESPONSE_CACHE[cache_key]
        ttl = get_cache_ttl(query)
        
        if time() - cached["timestamp"] < ttl:
            cached["hits"] += 1
            cache_stats["hits"] += 1
            cache_stats["saved_cost"] += 0.001
            logger.info(f"📦 CACHE HIT - Query matched (hits: {cached['hits']}, saved: ${cache_stats['saved_cost']:.4f})")
            return cached["response"]
        else:
            del RESPONSE_CACHE[cache_key]
            logger.info(f"📦 CACHE EXPIRED - Removing stale entry")
    
    cache_stats["misses"] += 1
    return None

def cache_response(query: str, response: str):
    """Store response in cache"""
    if len(RESPONSE_CACHE) >= MAX_CACHE_SIZE:
        oldest_key = min(RESPONSE_CACHE, key=lambda k: RESPONSE_CACHE[k]["timestamp"])
        del RESPONSE_CACHE[oldest_key]
        logger.info(f"📦 CACHE CLEANUP - Removed oldest entry")
    
    cache_key = get_cache_key(query)
    RESPONSE_CACHE[cache_key] = {
        "response": response,
        "timestamp": time(),
        "hits": 0,
        "original_query": query[:100]
    }
    logger.info(f"📦 CACHE STORED - Query cached (total entries: {len(RESPONSE_CACHE)})")

def clear_cache():
    """Clear all cache entries"""
    global RESPONSE_CACHE, cache_stats
    entries_cleared = len(RESPONSE_CACHE)
    RESPONSE_CACHE = {}
    cache_stats = {"hits": 0, "misses": 0, "saved_cost": 0.0}
    logger.info(f"📦 CACHE CLEARED - Removed {entries_cleared} entries")
    return entries_cleared

def get_cache_stats():
    """Get cache statistics"""
    return {
        "entries": len(RESPONSE_CACHE),
        "stats": cache_stats
    }
