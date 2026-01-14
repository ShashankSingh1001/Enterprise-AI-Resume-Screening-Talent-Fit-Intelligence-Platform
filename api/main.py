"""
Resume AI Platform - FastAPI Main Entry Point
Minimal version for Phase 1 testing
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Create FastAPI app
app = FastAPI(
    title="Resume AI Platform",
    description="Enterprise AI Resume Screening & Talent Fit Intelligence Platform",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will be configured properly in Phase 10
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint (for Docker health checks)
@app.get("/health")
async def health_check():
    """
    Health check endpoint for Docker and monitoring
    """
    return {
        "status": "healthy",
        "service": "resume-ai-api",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint - API information
    """
    return {
        "message": "Welcome to Resume AI Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# This allows the API to start without errors
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )