#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from monitor_tokens import start_monitoring, stop_monitoring, get_stats
import threading
import time

app = FastAPI(title="Token Monitor API", description="Real-time token usage monitoring")

# Global monitoring state
monitoring_active = False

@app.on_event("startup")
async def startup_event():
    """Start monitoring when API starts"""
    global monitoring_active
    if not monitoring_active:
        start_monitoring()
        monitoring_active = True
        print("🔍 Token monitoring started automatically!")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop monitoring when API shuts down"""
    global monitoring_active
    if monitoring_active:
        stop_monitoring()
        monitoring_active = False
        print("🛑 Token monitoring stopped!")

@app.get("/")
async def root():
    """Root endpoint with monitoring info"""
    return {
        "message": "Token Monitor API",
        "status": "running",
        "monitoring": monitoring_active,
        "endpoints": {
            "/stats": "Get current token usage statistics",
            "/start": "Start monitoring",
            "/stop": "Stop monitoring",
            "/summary": "Get detailed summary"
        }
    }

@app.get("/stats")
async def get_token_stats():
    """Get current token usage statistics"""
    try:
        stats = get_stats()
        return {
            "status": "success",
            "monitoring_active": monitoring_active,
            "statistics": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/start")
async def start_token_monitoring():
    """Start token monitoring"""
    global monitoring_active
    if monitoring_active:
        return {"message": "Monitoring is already active", "status": "already_running"}
    
    try:
        start_monitoring()
        monitoring_active = True
        return {"message": "Token monitoring started", "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting monitoring: {str(e)}")

@app.post("/stop")
async def stop_token_monitoring():
    """Stop token monitoring"""
    global monitoring_active
    if not monitoring_active:
        return {"message": "Monitoring is not active", "status": "not_running"}
    
    try:
        stop_monitoring()
        monitoring_active = False
        return {"message": "Token monitoring stopped", "status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping monitoring: {str(e)}")

@app.get("/summary")
async def get_detailed_summary():
    """Get detailed token usage summary"""
    try:
        from token_tracker import TokenTracker
        tracker = TokenTracker()
        
        # Get basic stats
        stats = get_stats()
        
        # Get detailed breakdown
        breakdown = {}
        for usage in tracker.usage_log:
            call_type = usage.call_type
            if call_type not in breakdown:
                breakdown[call_type] = {
                    "calls": 0,
                    "total_tokens": 0,
                    "total_cost": 0,
                    "avg_tokens_per_call": 0
                }
            
            breakdown[call_type]["calls"] += 1
            breakdown[call_type]["total_tokens"] += usage.total_tokens
            breakdown[call_type]["total_cost"] += usage.cost_estimate
        
        # Calculate averages
        for call_type in breakdown:
            calls = breakdown[call_type]["calls"]
            if calls > 0:
                breakdown[call_type]["avg_tokens_per_call"] = breakdown[call_type]["total_tokens"] / calls
        
        return {
            "status": "success",
            "monitoring_active": monitoring_active,
            "summary": {
                "statistics": stats,
                "breakdown": breakdown,
                "recent_calls": [
                    {
                        "timestamp": usage.timestamp,
                        "call_type": usage.call_type,
                        "function_name": usage.function_name,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                        "cost": usage.cost_estimate,
                        "model": usage.model
                    }
                    for usage in tracker.usage_log[-10:]  # Last 10 calls
                ]
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Token Monitor API...")
    print("📊 Available endpoints:")
    print("   GET  / - API info")
    print("   GET  /stats - Current statistics")
    print("   POST /start - Start monitoring")
    print("   POST /stop - Stop monitoring")
    print("   GET  /summary - Detailed summary")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8001) 