#!/usr/bin/env python3
"""Test script for configuration-driven query routing."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.config_loader import config_loader

def test_config_loader():
    """Test the configuration loader functionality."""
    print("🧪 Testing Configuration Loader")
    print("=" * 50)
    
    # Test database tables
    print("\n📊 Database Tables:")
    live_tables = config_loader.get_database_tables("live")
    common_tables = config_loader.get_database_tables("common")
    print(f"Live tables: {live_tables}")
    print(f"Common tables: {common_tables}")
    
    # Test keywords
    print("\n🔍 Keywords:")
    live_keywords = config_loader.get_keywords("live_only")
    common_keywords = config_loader.get_keywords("common_only")
    ledger_keywords = config_loader.get_keywords("requires_ledger")
    print(f"Live keywords: {live_keywords}")
    print(f"Common keywords: {common_keywords}")
    print(f"Ledger keywords: {ledger_keywords}")
    
    # Test query classification
    print("\n🎯 Query Classification:")
    test_queries = [
        "show last two order and their separate order amount and customer name his loyalty card of customer 1338787",
        "show my loyalty balance and transaction history",
        "what are my recent orders",
        "show my loyalty card number"
    ]
    
    for query in test_queries:
        needs_ledger = config_loader.needs_ledger_data(query)
        print(f"Query: '{query}'")
        print(f"  Needs ledger: {needs_ledger}")
    
    # Test fallback queries
    print("\n🔄 Fallback Queries:")
    fallback_queries = config_loader.get_fallback_queries()
    for name, query in fallback_queries.items():
        print(f"{name}: {query}")
    
    # Test both databases config
    print("\n🔗 Both Databases Config:")
    both_config = config_loader.get_both_databases_config()
    for key, value in both_config.items():
        print(f"{key}: {value}")
    
    print("\n✅ Configuration loader test completed!")

if __name__ == "__main__":
    test_config_loader() 