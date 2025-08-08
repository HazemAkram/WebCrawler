"""
DeepSeek AI Web Crawler
Copyright (c) 2025 Ayaz Mensyoğlu

This file is part of the DeepSeek AI Web Crawler project.
Licensed under the Apache License, Version 2.0.
See NOTICE file for additional terms and conditions.
"""


from flask import Flask, render_template, request, jsonify, send_file
import os
import asyncio
import threading
import time
import json
from werkzeug.utils import secure_filename
import tempfile
import shutil
from datetime import datetime
import secrets

# Import the crawling functions
from main import crawl_from_sites_csv, set_log_callback, log_message
from config import DEFAULT_CONFIG, ENV_VARS

app = Flask(__name__)

# SECURITY: Generate a random secret key instead of using a hardcoded one
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# SECURITY: Disable debug mode in production
DEBUG_MODE = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for crawling state
crawling_status = {
    'is_running': False,
    'current_site': 0,
    'total_sites': 0,
    'current_page': 1,
    'total_venues': 0,
    'logs': [],
    'start_time': None,
    'stop_requested': False
}

# Thread-safe logging
import queue
log_queue = queue.Queue()

def log_message(message, level="INFO"):
    """Add a log message to the queue"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        'timestamp': timestamp,
        'level': level,
        'message': message
    }
    log_queue.put(log_entry)
    crawling_status['logs'].append(log_entry)
    
    # Keep only last 1000 logs
    if len(crawling_status['logs']) > 1000:
        crawling_status['logs'] = crawling_status['logs'][-1000:]

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

@app.route('/')
def index():
    """Main page with the web interface"""
    return render_template('index.html', 
                         config=DEFAULT_CONFIG,
                         env_vars=ENV_VARS)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Count sites in CSV
        import csv
        try:
            with open(filepath, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                sites = list(reader)
                total_sites = len(sites)
        except Exception as e:
            return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'total_sites': total_sites
        })
    
    return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

def run_crawler_async(csv_filepath, api_key, model):
    """Run the crawler in a separate thread"""
    global crawling_status
    
    try:
        # Set up initial status
        crawling_status['is_running'] = True
        crawling_status['stop_requested'] = False
        crawling_status['start_time'] = datetime.now()
        crawling_status['logs'] = []
        
        # Set up logging callback
        def web_log_callback(message, level):
            log_message(message, level)
        
        set_log_callback(web_log_callback)
        
        # Set up status callback
        def status_callback(status_update):
            for key, value in status_update.items():
                crawling_status[key] = value
        
        # Set up stop request callback
        def stop_requested_callback():
            return crawling_status['stop_requested']
        
        log_message("Starting web crawler...", "INFO")
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the crawler
        loop.run_until_complete(crawl_from_sites_csv(
            input_file=csv_filepath,
            api_key=api_key,
            model=model,
            status_callback=status_callback,
            stop_requested_callback=stop_requested_callback
        ))
        
        if not crawling_status['stop_requested']:
            log_message("Crawling completed successfully!", "SUCCESS")
        else:
            log_message("Crawling stopped by user request.", "WARNING")
            
    except Exception as e:
        log_message(f"Error during crawling: {str(e)}", "ERROR")
    finally:
        crawling_status['is_running'] = False
        crawling_status['stop_requested'] = False
        loop.close()

@app.route('/start_crawling', methods=['POST'])
def start_crawling():
    """Start the crawling process"""
    global crawling_status
    
    if crawling_status['is_running']:
        return jsonify({'error': 'Crawling is already running'}), 400
    
    data = request.get_json()
    csv_filepath = data.get('csv_filepath')
    api_key = data.get('api_key')
    model = data.get('model', DEFAULT_CONFIG['default_model'])
    
    if not csv_filepath or not os.path.exists(csv_filepath):
        return jsonify({'error': 'Invalid CSV file path'}), 400
    
    if not api_key:
        return jsonify({'error': 'API key is required'}), 400
    
    # Start crawling in a separate thread
    thread = threading.Thread(
        target=run_crawler_async,
        args=(csv_filepath, api_key, model)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Crawling started'})

@app.route('/stop_crawling', methods=['POST'])
def stop_crawling():
    """Stop the crawling process"""
    global crawling_status
    
    if not crawling_status['is_running']:
        return jsonify({'error': 'No crawling process running'}), 400
    
    crawling_status['stop_requested'] = True
    log_message("Stop request received. Finishing current task...", "WARNING")
    
    return jsonify({'success': True, 'message': 'Stop request sent'})

@app.route('/status')
def get_status():
    """Get current crawling status"""
    global crawling_status
    
    # Calculate elapsed time
    elapsed_time = None
    if crawling_status['start_time']:
        elapsed_time = (datetime.now() - crawling_status['start_time']).total_seconds()
    
    return jsonify({
        'is_running': crawling_status['is_running'],
        'current_site': crawling_status['current_site'],
        'total_sites': crawling_status['total_sites'],
        'current_page': crawling_status['current_page'],
        'total_venues': crawling_status['total_venues'],
        'elapsed_time': elapsed_time,
        'logs': crawling_status['logs'][-50:],  # Return last 50 logs
        'stop_requested': crawling_status['stop_requested']
    })

@app.route('/logs')
def get_logs():
    """Get all logs"""
    return jsonify({'logs': crawling_status['logs']})

@app.route('/download_output')
def download_output():
    """Download the output folder as a zip file"""
    import zipfile
    
    output_folder = "output"
    if not os.path.exists(output_folder):
        return jsonify({'error': 'No output folder found'}), 404
    
    # Create a temporary zip file
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    
    with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_folder)
                zipf.write(file_path, arcname)
    
    return send_file(
        temp_zip.name,
        as_attachment=True,
        download_name=f'crawler_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
        mimetype='application/zip'
    )

if __name__ == '__main__':
    # SECURITY: Only bind to localhost (127.0.0.1) instead of 0.0.0.0
    # This prevents external access from the internet
    app.run(debug=DEBUG_MODE, host='127.0.0.1', port=5000) 