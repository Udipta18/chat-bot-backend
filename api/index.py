"""
Vercel Serverless Entry Point
"""
import sys
import os
from pathlib import Path

# Get the backend directory (parent of api directory)
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Now import the app
from app.main import app

# Export for Vercel
handler = app
