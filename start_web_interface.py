"""
DeepSeek AI Web Crawler
Copyright (c) 2025 Ayaz Mensyoğlu

This file is part of the DeepSeek AI Web Crawler project.
Licensed under the Apache License, Version 2.0.
See NOTICE file for additional terms and conditions.
"""


#!/usr/bin/env python3
"""
Startup script for the DeepSeek AI Web Crawler Web Interface
"""

import os
import sys
import subprocess
import webbrowser
import time

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import crawl4ai
        import dotenv
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['uploads', 'output', 'templates']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def main():
    print("🚀 Starting DeepSeek AI Web Crawler Web Interface")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create necessary directories
    create_directories()
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ app.py not found. Please ensure you're in the correct directory.")
        return
    
    print("\n🌐 Starting web server...")
    print("📱 The web interface will open automatically in your browser")
    print("🔗 Manual access: http://localhost:5000")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:5000')
        except:
            pass
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the Flask app
    try:
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting server: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 