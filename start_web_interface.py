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
import secrets

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

def setup_security():
    """Set up security environment variables"""
    # Generate a secure secret key if not already set
    if not os.environ.get('FLASK_SECRET_KEY'):
        secret_key = secrets.token_hex(32)
        os.environ['FLASK_SECRET_KEY'] = secret_key
        print("🔐 Generated secure secret key")
    
    # Set debug mode to False for production
    if not os.environ.get('FLASK_DEBUG'):
        os.environ['FLASK_DEBUG'] = 'False'
        print("🛡️ Debug mode disabled for security")

def security_warning():
    """Display security warning"""
    print("\n" + "="*60)
    print("🚨 SECURITY WARNING 🚨")
    print("="*60)
    print("This web interface is configured for EXTERNAL ACCESS.")
    print("⚠️  External access is enabled - ensure proper security measures are in place.")
    print("\n🔒 Current security settings:")
    print("   • Host: 65.108.122.8 (external access enabled)")
    print("   • Debug mode: Disabled")
    print("   • Secret key: Auto-generated")
    print("\n🛡️ For production deployment, consider:")
    print("   • Adding authentication")
    print("   • Using HTTPS")
    print("   • Implementing rate limiting")
    print("   • Using a reverse proxy (nginx)")
    print("   • Setting up a firewall")
    print("="*60)

def main():
    print("🚀 Starting DeepSeek AI Web Crawler Web Interface")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create necessary directories
    create_directories()
    
    # Set up security
    setup_security()
    
    # Display security warning
    security_warning()
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ app.py not found. Please ensure you're in the correct directory.")
        return
    
    print("\n🌐 Starting web server...")
    print("📱 The web interface will open automatically in your browser")
    print("🔗 Access: http://65.108.122.8:5000")
    print("🌐 External access: ENABLED")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://65.108.122.8:5000')
        except:
            pass
    
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