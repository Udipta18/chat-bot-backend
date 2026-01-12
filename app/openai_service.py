"""
OpenAI client and chat service
"""
import json
import logging
from typing import Optional
from openai import OpenAI
from fastapi import HTTPException

from app.config import OPENAI_API_KEY, MODEL_NAME, calculate_cost
from app.tools import TOOL_FUNCTIONS, OPENAI_TOOLS

logger = logging.getLogger(__name__)

# ============================================
# OPENAI CLIENT (LAZY INITIALIZATION)
# ============================================

_client: Optional[OpenAI] = None

def get_openai_client() -> OpenAI:
    """Get or create OpenAI client"""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY not set. Please add it to your .env file"
            )
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

# ============================================
# CHAT SERVICE
# ============================================

def chat_with_openai(message: str, history: list) -> tuple[str, float]:
    """
    Process message using OpenAI with function calling.
    Returns: (response_text, total_cost)
    """
    client = get_openai_client()
    total_cost = 0.0
    
    # Build messages
    messages = [
        {
            "role": "system",
            "content": """You are a helpful travel assistant for VIP Travels. 
            You help users with tour bookings, bus fares, and travel information.
            Use the available tools to provide accurate information.
            Keep responses concise and friendly.
            If asked about topics not related to travel, politely redirect to travel topics."""
        }
    ] + history + [{"role": "user", "content": message}]
    
    # Call OpenAI with function calling
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=OPENAI_TOOLS,
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
            
            if function_name in TOOL_FUNCTIONS:
                func = TOOL_FUNCTIONS[function_name]
                result = func(**function_args) if function_args else func()
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
        
        return response_text, total_cost
    
    # No function call, return direct response
    response_text = response_message.content
    
    logger.info(f"💬 Direct response: {response_text[:100]}")
    logger.info(f"💰 Cost: {response.usage.prompt_tokens} input + {response.usage.completion_tokens} output = ${total_cost:.6f}")
    
    return response_text, total_cost
