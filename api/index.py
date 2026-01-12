"""
Vercel Serverless Entry Point
Simply imports the app from the modular structure
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import app module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the FastAPI app from our modular structure
from app.main import app
