#!/bin/bash

# Voice Chatbot Backend Startup Script

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY is not set!"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY=sk-your-api-key-here"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ OpenAI API key found"
echo "🚀 Starting backend server..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Start uvicorn
uvicorn app.main:app --reload
