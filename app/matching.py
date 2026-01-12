"""
Direct tool matching for common queries
Matches keywords to tools WITHOUT calling OpenAI (FREE!)
"""
import re
import logging
from typing import Optional, Tuple, Dict, Any

from app.tools import TOOL_FUNCTIONS

logger = logging.getLogger(__name__)

# ============================================
# KEYWORD PATTERNS FOR DIRECT MATCHING
# ============================================

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

# ============================================
# STATISTICS
# ============================================

direct_match_stats = {
    "matches": 0,
    "api_calls_saved": 0,
    "cost_saved": 0.0
}

# ============================================
# MATCHING FUNCTION
# ============================================

def match_direct_tool(query: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Match query to a tool using keywords (FREE - no OpenAI call).
    Returns: (tool_name, params) or (None, None)
    """
    query_lower = query.lower().strip()
    
    for tool_name, config in DIRECT_TOOL_PATTERNS.items():
        keywords = config.get("keywords", [])
        
        if any(kw in query_lower for kw in keywords):
            # Special handling for calculator
            if tool_name == "calculate":
                operators = config.get("operators", [])
                if any(op in query_lower for op in operators):
                    numbers = re.findall(r'\d+\.?\d*', query)
                    if len(numbers) >= 2:
                        for op in ['+', '-', '*', '/']:
                            if op in query:
                                return tool_name, {"expression": f"{numbers[0]} {op} {numbers[1]}"}
                        op_map = {"plus": "+", "minus": "-", "times": "*", "divided": "/", "multiply": "*"}
                        for word, op in op_map.items():
                            if word in query_lower:
                                return tool_name, {"expression": f"{numbers[0]} {op} {numbers[1]}"}
            else:
                return tool_name, None
    
    return None, None

def execute_direct_tool(tool_name: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Execute a tool directly without OpenAI"""
    if tool_name not in TOOL_FUNCTIONS:
        return None
    
    func = TOOL_FUNCTIONS[tool_name]
    
    # Update stats
    direct_match_stats["matches"] += 1
    direct_match_stats["api_calls_saved"] += 1
    direct_match_stats["cost_saved"] += 0.001
    
    logger.info(f"🆓 FREE - Direct tool match: {tool_name}")
    
    if params:
        result = func(**params)
    else:
        result = func()
    
    logger.info(f"🆓 Direct tool result: {result[:100]}...")
    logger.info(f"💚 API calls saved: {direct_match_stats['api_calls_saved']}, Total saved: ${direct_match_stats['cost_saved']:.4f}")
    
    return result

def get_direct_match_stats():
    """Get direct matching statistics"""
    return direct_match_stats
