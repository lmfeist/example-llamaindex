#!/usr/bin/env python3
"""
Test script to demonstrate error handling and stack trace logging.
This script will start the server and make requests that trigger errors
to show that stack traces are properly logged to the console.
"""

import requests
import time
import subprocess
import signal
import sys
import os
from threading import Thread

def start_server():
    """Start the FastAPI server in a subprocess."""
    env = os.environ.copy()
    env['PORT'] = '8001'  # Use different port for testing
    
    # Start the server
    process = subprocess.Popen([
        sys.executable, 'app.py'
    ], env=env, cwd='/workspace')
    
    # Wait a bit for server to start
    time.sleep(3)
    
    return process

def test_error_scenarios():
    """Test various error scenarios to trigger stack traces."""
    base_url = "http://localhost:8001"
    
    print("Testing error scenarios to verify stack trace logging...")
    
    # Test 1: Invalid URL format
    print("\n1. Testing invalid URL format...")
    try:
        response = requests.post(
            f"{base_url}/summarize",
            json={"url": "not-a-valid-url"},
            timeout=5
        )
        print(f"Response status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Request failed: {e}")
    
    # Test 2: Non-existent domain
    print("\n2. Testing non-existent domain...")
    try:
        response = requests.post(
            f"{base_url}/summarize", 
            json={"url": "https://thisdoesnotexist12345.com"},
            timeout=10
        )
        print(f"Response status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Request failed: {e}")
    
    # Test 3: Malformed request
    print("\n3. Testing malformed request...")
    try:
        response = requests.post(
            f"{base_url}/summarize",
            json={"invalid_field": "test"},
            timeout=5
        )
        print(f"Response status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Request failed: {e}")

def main():
    """Main function to run the test."""
    print("Starting error handling test...")
    
    # Start the server
    print("Starting server...")
    server_process = start_server()
    
    try:
        # Run tests
        test_error_scenarios()
        
        print("\n" + "="*60)
        print("Tests completed!")
        print("Check the server console output above for detailed stack traces.")
        print("The server should have logged full stack traces for any errors encountered.")
        print("="*60)
        
    finally:
        # Clean up - stop the server
        print("\nStopping test server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main()