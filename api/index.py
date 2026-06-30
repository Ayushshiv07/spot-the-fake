"""
api/index.py — Vercel serverless entry point.

Serves the full Spot the Fake Photo 3D Dashboard on Vercel.
Imports from the parent directory where all source files live.
"""

import sys
import os

# Add parent directory to path so we can import predict, features etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app object from app.py in the project root
from app import app

# Vercel expects the WSGI callable to be named 'app'
# app is already the Flask instance — this is all Vercel needs!
