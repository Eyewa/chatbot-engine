#!/usr/bin/env python3

import time
import requests
import json
from datetime import datetime
import os

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def create_bar_chart(data, max_width=50):
    """Create ASCII bar chart"""
    if not data:
        return "No data available"
    
    max_value = max(data.values()) if data.values() else 1
    chart = []
    
    for label, value in data.items():
        bar_length = int((value / max_value) * max_width) if max_value > 0 else 0
        bar = "█" * bar_length
        chart.append(f"{label:<20} {bar} {value}")
    
    return "\n".join(chart)

def format_duration(seconds):
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_monitor_data():
    """Get data from token monitor API"""
    try:
        response = requests.get("http://localhost:8000/monitor/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def get_detailed_summary():
    """Get detailed summary from API"""
    try:
        response = requests.get("http://localhost:8000/monitor/summary", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def display_dashboard():
    """Display the main dashboard"""
    clear_screen()
    
    print("🔍 TOKEN USAGE DASHBOARD")
    print("=" * 60)
    print(f"📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get basic stats
    stats_data = get_monitor_data()
    if not stats_data:
        print("❌ Cannot connect to Token Monitor API")
        print("   Make sure the chatbot app is running on http://localhost:8000")
        print("   Run: python3 main.py")
        return
    
    stats = stats_data.get("statistics", {})
    monitoring_active = stats_data.get("monitoring_active", False)
    
    # Status indicator
    status_emoji = "🟢" if monitoring_active else "🔴"
    print(f"{status_emoji} Monitoring Status: {'ACTIVE' if monitoring_active else 'INACTIVE'}")
    print()
    
    # Key metrics
    print("📊 KEY METRICS:")
    print(f"   Total Calls: {stats.get('total_calls', 0)}")
    print(f"   Total Tokens: {stats.get('total_tokens', 0):,}")
    print(f"   Total Cost: ${stats.get('total_cost', 0):.4f}")
    print(f"   Calls/Minute: {stats.get('calls_per_minute', 0):.2f}")
    print(f"   Rate Limit Usage: {stats.get('rate_limit_usage', 0):.1f}%")
    print()
    
    # Rate limit warning
    rate_limit_usage = stats.get('rate_limit_usage', 0)
    if rate_limit_usage > 80:
        print("⚠️  WARNING: Rate limit usage is high!")
        print(f"   Current usage: {rate_limit_usage:.1f}% of 30,000 TPM limit")
        print()
    elif rate_limit_usage > 50:
        print("⚠️  Rate limit usage is moderate")
        print(f"   Current usage: {rate_limit_usage:.1f}% of 30,000 TPM limit")
        print()
    
    # Get detailed breakdown
    summary_data = get_detailed_summary()
    if summary_data and "summary" in summary_data:
        breakdown = summary_data["summary"].get("breakdown", {})
        
        if breakdown:
            print("📈 USAGE BREAKDOWN:")
            
            # Create bar chart data
            chart_data = {}
            for call_type, data in breakdown.items():
                chart_data[call_type] = data.get("calls", 0)
            
            print(create_bar_chart(chart_data))
            print()
            
            # Detailed breakdown
            print("📋 DETAILED BREAKDOWN:")
            for call_type, data in breakdown.items():
                calls = data.get("calls", 0)
                total_tokens = data.get("total_tokens", 0)
                total_cost = data.get("total_cost", 0)
                avg_tokens = data.get("avg_tokens_per_call", 0)
                
                print(f"   {call_type.upper()}:")
                print(f"     Calls: {calls}")
                print(f"     Total Tokens: {total_tokens:,}")
                print(f"     Total Cost: ${total_cost:.4f}")
                print(f"     Avg Tokens/Call: {avg_tokens:.1f}")
                print()
        
        # Recent calls
        recent_calls = summary_data["summary"].get("recent_calls", [])
        if recent_calls:
            print("🕒 RECENT CALLS (Last 5):")
            for call in recent_calls[-5:]:
                timestamp = call.get("timestamp", "")
                call_type = call.get("call_type", "")
                function = call.get("function_name", "")
                total_tokens = call.get("total_tokens", 0)
                cost = call.get("cost", 0)
                
                print(f"   {timestamp} | {call_type} | {function} | "
                      f"Tokens: {total_tokens} | Cost: ${cost:.4f}")
            print()
    
    print("=" * 60)
    print("💡 Tips:")
    print("   • Monitor rate limit usage to avoid hitting limits")
    print("   • Consider caching for repeated queries")
    print("   • Use the /summary endpoint for detailed analysis")
    print("   • Press Ctrl+C to exit")
    print("=" * 60)

def main():
    """Main dashboard loop"""
    print("🚀 Starting Token Usage Dashboard...")
    print("📊 Connecting to Token Monitor API...")
    print()
    
    try:
        while True:
            display_dashboard()
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        print("👋 Goodbye!")

if __name__ == "__main__":
    main() 