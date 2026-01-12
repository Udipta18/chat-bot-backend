"""
Guardrails for content moderation
Blocks inappropriate content before API calls
"""
import logging

logger = logging.getLogger(__name__)

# ============================================
# BLOCKED CONTENT
# ============================================

# Foul/inappropriate words list
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

# ============================================
# GUARDRAIL FUNCTION
# ============================================

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
