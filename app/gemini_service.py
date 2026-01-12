"""
Google Gemini client and chat service (using new google-genai SDK)
"""
import logging
from google import genai
from google.genai import types
from typing import Optional, List, Dict, Any
from fastapi import HTTPException

from app.config import GOOGLE_API_KEY, GEMINI_MODEL_NAME
from app.tools import OPENAI_TOOLS, TOOL_FUNCTIONS

logger = logging.getLogger(__name__)

# ============================================
# GEMINI CLIENT (LAZY INITIALIZATION)
# ============================================

_client = None

def get_gemini_client():
    """Get or create Gemini client instance"""
    global _client
    if _client is None:
        if not GOOGLE_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GOOGLE_API_KEY not set. Please add it to your .env file"
            )
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client

def convert_tools_to_gemini_format(openai_tools: List[Dict]) -> types.Tool:
    """Convert OpenAI tool definitions to Gemini format"""
    function_declarations = []
    
    for tool in openai_tools:
        if tool["type"] == "function":
            func = tool["function"]
            
            # Convert parameters
            properties = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])
            
            # Gemini expects a slightly different schema
            # But the structure is similar enough for simple tools
            # We map OpenAI JSON schema to Gemini properties
            
            gemini_properties = {}
            for name, prop in properties.items():
                gemini_properties[name] = {
                    "type": prop.get("type").upper(), # STRING, INTEGER, etc.
                    "description": prop.get("description", "")
                }
            
            function_declarations.append({
                "name": func["name"],
                "description": func["description"],
                "parameters": {
                    "type": "OBJECT",
                    "properties": gemini_properties,
                    "required": required
                }
            })
            
    return types.Tool(function_declarations=function_declarations)

# ============================================
# CHAT SERVICE
# ============================================

def chat_with_gemini(message: str, history: list) -> tuple[str, float]:
    """
    Process message using Google Gemini with function calling.
    Returns: (response_text, estimated_cost)
    """
    try:
        client = get_gemini_client()
        
        # Configure tool
        tools = convert_tools_to_gemini_format(OPENAI_TOOLS)
        
        system_instruction = """You are a helpful travel assistant for VIP Travels. 
        You help users with tour bookings, bus fares, and travel information.
        Use the available tools to provide accurate information.
        Keep responses concise and friendly.
        If asked about topics not related to travel, politely redirect to travel topics."""
        
        # Build prompt
        contents = [types.Content(role="user", parts=[types.Part(text=message)])]
        
        config = types.GenerateContentConfig(
            tools=[tools],
            system_instruction=system_instruction,
            temperature=0.7,
            max_output_tokens=300
        )
        
        # Send request
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME if GEMINI_MODEL_NAME else "gemini-2.0-flash", # Use standard model
            contents=contents,
            config=config,
        )
        
        # Initialize output
        response_text = ""
        total_cost = 0.0001
        
        # Handle function calls
        # Gemini 1.0+ SDK response structure handling
        if response.function_calls:
            # We have tool calls
             for call in response.function_calls:
                fn_name = call.name
                fn_args = call.args
                
                logger.info(f"🔧 Gemini wants to call tool: {fn_name} with args: {fn_args}")
                
                if fn_name in TOOL_FUNCTIONS:
                    func = TOOL_FUNCTIONS[fn_name]
                    # Convert args to dict if needed, usually already dict
                    result = func(**fn_args)
                else:
                    result = f"Unknown function: {fn_name}"
                    
                logger.info(f"✅ Tool result: {result[:100]}...")
                
                # Send tool response back (multi-turn conversation)
                # We need to construct the history: User Query -> Model Call -> User Result -> Model Answer
                # This is tricky in stateless, simpler to make a new call with tool result
                
                # Simplified: Just append tool result and ask for final answer
                # For proper function calling loop in Gemini, we use chat sessions usually
                # Here we simulate by just getting the result and returning it (or feeding back)
                
                # Let's try to just return the result directly if it's the final answer we want
                # Or create a 2nd turn.
                
                response_text = str(result) # For now, just return tool output
                
                # Ideally we send it back to model:
                # chat.send_message(types.Part(function_response=...))
                
                # Re-prompt model with result
                new_contents = [
                    types.Content(role="user", parts=[types.Part(text=message)]),
                    types.Content(role="model", parts=[types.Part(function_call=call)]),
                    types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(name=fn_name, response={"result": result}))])
                ]
                
                final_response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=new_contents,
                    config=config
                )
                response_text = final_response.text

        else:
            response_text = response.text
            
        logger.info(f"💬 Gemini Response: {response_text[:100]}")
        return response_text, total_cost

    except Exception as e:
        logger.error(f"❌ Gemini Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gemini Error: {str(e)}")
