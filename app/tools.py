"""
Tool functions for the chatbot
Add your custom tools here
"""
from datetime import datetime

# ============================================
# TOOL FUNCTIONS
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
        allowed_chars = set('0123456789+-*/.(). ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression, {"__builtins__": {}}, {})
            return f"The result of {expression} is {result}"
        else:
            return "I can only calculate basic math expressions with numbers and +, -, *, /"
    except Exception as e:
        return f"I couldn't calculate that expression: {str(e)}"

def get_recent_tours() -> str:
    """Returns a list of available travel tours"""
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
    """Returns bus fares for different bus types"""
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
    """Returns contact information"""
    return """📞 Contact Information:

• Phone: +91 98765 43210
• WhatsApp: +91 98765 43210
• Email: info@viptravels.com
• Office: 123 Travel Street, New Delhi, India 110001

Office Hours: Mon-Sat, 9:00 AM - 6:00 PM"""

def get_booking_info() -> str:
    """Returns booking information"""
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
# TOOL REGISTRY
# ============================================

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
# OPENAI TOOL DEFINITIONS (for function calling)
# ============================================

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time. Use this when user asks about time.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get today's date. Use this when user asks about today's date.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to calculate"
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
            "description": "Get list of available travel tours, vacation packages, and trips.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_bus_fares",
            "description": "Get bus ticket prices and fares for different bus types.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_contact_info",
            "description": "Get contact information including phone, email, and address.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_booking_info",
            "description": "Get information about how to book tours or buses.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]
