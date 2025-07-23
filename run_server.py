#!/usr/bin/env python3
"""
Simple script to start the ART HTTP Training Service.
"""

import uvicorn
from src.art_http.config import config

if __name__ == "__main__":
    print("🚀 Starting ART HTTP Training Service...")
    print(f"   Host: {config.server.host}")
    print(f"   Port: {config.server.port}")
    print(f"   Backend: {config.art.backend}")
    print(f"   OpenAI configured: {bool(config.openai.api_key)}")
    print("=" * 50)
    
    uvicorn.run(
        "src.art_http.api:app",
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
        reload=True
    )