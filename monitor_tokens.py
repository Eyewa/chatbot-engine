#!/usr/bin/env python3

import time
import json
import threading
from datetime import datetime, timedelta
from token_tracker import TokenTracker

# Import the print_summary function correctly
def print_summary():
    """Print summary using the tracker's method"""
    tracker = TokenTracker()
    tracker.print_summary()

class TokenMonitor:
    def __init__(self):
        self.tracker = TokenTracker()
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        
    def start_monitoring(self):
        """Start real-time token monitoring"""
        if self.monitoring:
            print("⚠️  Monitoring is already active!")
            return
            
        self.monitoring = True
        self.start_time = datetime.now()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print("🔍 Token monitoring started!")
        print("📊 Press Ctrl+C to stop monitoring and see summary")
        print("=" * 60)
        
    def stop_monitoring(self):
        """Stop monitoring and show summary"""
        if not self.monitoring:
            print("⚠️  Monitoring is not active!")
            return
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            
        print("\n" + "=" * 60)
        print("🛑 Monitoring stopped!")
        self._print_session_summary()
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        last_count = 0
        while self.monitoring:
            current_count = len(self.tracker.usage_log)
            
            if current_count > last_count:
                # New calls detected
                new_calls = current_count - last_count
                print(f"🆕 {new_calls} new LLM call(s) detected!")
                self._print_recent_usage()
                last_count = current_count
                
            time.sleep(2)  # Check every 2 seconds
            
    def _print_recent_usage(self):
        """Print recent token usage"""
        if not self.tracker.usage_log:
            return
            
        recent = self.tracker.usage_log[-3:]  # Last 3 calls
        print("📈 Recent calls:")
        for usage in recent:
            print(f"   • {usage.call_type} | {usage.function_name} | "
                  f"Input: {usage.input_tokens} | Output: {usage.output_tokens} | "
                  f"Total: {usage.total_tokens} | Cost: ${usage.cost_estimate:.4f}")
        print()
        
    def _print_session_summary(self):
        """Print session summary"""
        if not self.start_time:
            return
            
        duration = datetime.now() - self.start_time
        total_calls = len(self.tracker.usage_log)
        
        print(f"⏱️  Session Duration: {duration}")
        print(f"📊 Total Calls: {total_calls}")
        
        if total_calls > 0:
            calls_per_minute = total_calls / (duration.total_seconds() / 60)
            print(f"🚀 Calls per minute: {calls_per_minute:.2f}")
            
            # Estimate rate limit usage
            estimated_tokens = sum(u.total_tokens for u in self.tracker.usage_log)
            rate_limit_usage = (estimated_tokens / 30000) * 100
            print(f"⚠️  Estimated rate limit usage: {rate_limit_usage:.1f}%")
            
        print_summary()
        
    def get_live_stats(self):
        """Get current statistics"""
        if not self.tracker.usage_log:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost": 0,
                "calls_per_minute": 0,
                "rate_limit_usage": 0
            }
            
        total_calls = len(self.tracker.usage_log)
        total_tokens = sum(u.total_tokens for u in self.tracker.usage_log)
        total_cost = sum(u.cost_estimate for u in self.tracker.usage_log)
        
        if self.start_time:
            duration = datetime.now() - self.start_time
            calls_per_minute = total_calls / (duration.total_seconds() / 60) if duration.total_seconds() > 0 else 0
        else:
            calls_per_minute = 0
            
        rate_limit_usage = (total_tokens / 30000) * 100
        
        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "calls_per_minute": calls_per_minute,
            "rate_limit_usage": rate_limit_usage
        }

# Global monitor instance
monitor = TokenMonitor()

def start_monitoring():
    """Start token monitoring"""
    monitor.start_monitoring()

def stop_monitoring():
    """Stop token monitoring"""
    monitor.stop_monitoring()

def get_stats():
    """Get current statistics"""
    return monitor.get_live_stats()

if __name__ == "__main__":
    print("🔍 Token Usage Monitor")
    print("=" * 40)
    
    try:
        monitor.start_monitoring()
        
        # Keep the main thread alive
        while monitor.monitoring:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        monitor.stop_monitoring()
    except Exception as e:
        print(f"❌ Error: {e}")
        monitor.stop_monitoring() 