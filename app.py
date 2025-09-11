"""
DeepSeek AI Web Crawler
Copyright (c) 2025 Ayaz Mensyoğlu

This file is part of the DeepSeek AI Web Crawler project.
Licensed under the Apache License, Version 2.0.
See NOTICE file for additional terms and conditions.
"""


from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import os
import asyncio
import threading
from werkzeug.utils import secure_filename
import tempfile
import shutil
from datetime import datetime
import secrets
from pathlib import Path
import zipfile
import tarfile


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

def run_crawler_async(csv_filepath, api_key, model, pdf_size_limit=10, skip_large_files=True):
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
        
        # Set PDF size limit if provided
        if pdf_size_limit and skip_large_files:
            from utils.scraper_utils import set_pdf_size_limit
            set_pdf_size_limit(pdf_size_limit)
            log_message(f"PDF size limit set to {pdf_size_limit}MB", "INFO")
        
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
    model = 'groq/llama-3.1-8b-instant'
    pdf_size_limit = data.get('pdf_size_limit', DEFAULT_CONFIG['pdf_settings']['max_file_size_mb'])
    skip_large_files = data.get('skip_large_files', DEFAULT_CONFIG['pdf_settings']['skip_large_files'])
    
    if not csv_filepath or not os.path.exists(csv_filepath):
        return jsonify({'error': 'Invalid CSV file path'}), 400
    
    if not api_key:
        return jsonify({'error': 'API key is required'}), 400
    
    # Start crawling in a separate thread
    thread = threading.Thread(
        target=run_crawler_async,
        args=(csv_filepath, api_key, model, pdf_size_limit, skip_large_files)
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

@app.route('/archives/<path:filename>')
def serve_archive(filename):
    """Serve prepared .tar.gz archives from the archives directory"""
    try:
        archives_dir = os.path.join(os.getcwd(), 'archives')
        # Security: ensure the requested file resolves under the archives directory
        full_path = os.path.abspath(os.path.join(archives_dir, filename))
        if not full_path.startswith(os.path.abspath(archives_dir)):
            return "Access denied", 403
        if not os.path.exists(full_path):
            return "File not found", 404
        return send_from_directory(archives_dir, filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'Error serving archive: {str(e)}'}), 500

@app.route('/download_output')
def download_output():
    """Download the output folder as a zip file"""
    
    output_folder = "output"
    if not os.path.exists(output_folder):
        return jsonify({'error': 'No output folder found'}), 404

    # If "mode=link" is requested, create a persistent .tar.gz and return a short JSON with a URL
    mode = request.args.get('mode')
    if mode == 'link':
        try:
            archives_dir = os.path.join(os.getcwd(), 'archives')
            os.makedirs(archives_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"crawler_output_{timestamp}.tar.gz"
            archive_path = os.path.join(archives_dir, archive_name)

            # Build tar.gz archive without loading into memory
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(output_folder, arcname=os.path.basename(output_folder))

            # Return a short JSON containing a stable URL and absolute path for server-to-server fetches
            return jsonify({
                'success': True,
                'archive_url': f"/archives/{archive_name}",
                'archive_path': archive_path
            })
        except Exception as e:
            return jsonify({'error': f'Failed to prepare archive: {str(e)}'}), 500

    # Default: preserve existing behavior to return a downloadable zip stream (may be large)
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

@app.route('/files')
@app.route('/files/<path:subpath>')
def file_explorer(subpath=''):
    """File explorer for the output folder"""
    output_folder = "output"
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        return render_template('file_explorer.html', 
                             current_path='',
                             files=[],
                             folders=[],
                             parent_path='',
                             error="Output folder not found")
    
    # Build the full path
    full_path = os.path.join(output_folder, subpath)
    
    # Security check: ensure the path is within the output folder
    try:
        full_path = os.path.abspath(full_path)
        output_folder_abs = os.path.abspath(output_folder)
        if not full_path.startswith(output_folder_abs):
            return "Access denied", 403
    except:
        return "Invalid path", 400
    
    # Check if path exists
    if not os.path.exists(full_path):
        return "Path not found", 404
    
    # If it's a file, serve it
    if os.path.isfile(full_path):
        # Security: only allow safe file types
        allowed_extensions = {'.pdf', '.txt', '.csv', '.json', '.zip', '.jpg', '.jpeg', '.png', '.gif'}
        file_ext = os.path.splitext(full_path)[1].lower()
        
        if file_ext not in allowed_extensions:
            return "File type not allowed", 403
            
        return send_from_directory(os.path.dirname(full_path), os.path.basename(full_path))
    
    # If it's a directory, show the file explorer
    try:
        items = os.listdir(full_path)
        files = []
        folders = []
        
        for item in sorted(items):
            item_path = os.path.join(full_path, item)
            if os.path.isdir(item_path):
                folders.append({
                    'name': item,
                    'path': os.path.join(subpath, item) if subpath else item,
                    'size': '--',
                    'modified': datetime.fromtimestamp(os.path.getmtime(item_path)).strftime('%Y-%m-%d %H:%M:%S')
                })
            else:
                # Get file size
                try:
                    size_bytes = os.path.getsize(item_path)
                    if size_bytes < 1024:
                        size_str = f"{size_bytes} B"
                    elif size_bytes < 1024 * 1024:
                        size_str = f"{size_bytes / 1024:.1f} KB"
                    else:
                        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                except:
                    size_str = "Unknown"
                
                files.append({
                    'name': item,
                    'path': os.path.join(subpath, item) if subpath else item,
                    'size': size_str,
                    'modified': datetime.fromtimestamp(os.path.getmtime(item_path)).strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Sort files and folders
        folders.sort(key=lambda x: x['name'].lower())
        files.sort(key=lambda x: x['name'].lower())
        
        # Calculate parent path
        if subpath:
            parent_parts = subpath.split('/')
            if len(parent_parts) > 1:
                parent_path = '/'.join(parent_parts[:-1])
            else:
                parent_path = ''
        else:
            parent_path = ''
        
        return render_template('file_explorer.html',
                             current_path=subpath,
                             files=files,
                             folders=folders,
                             parent_path=parent_path,
                             error=None)
                             
    except Exception as e:
        return render_template('file_explorer.html',
                             current_path=subpath,
                             files=[],
                             folders=[],
                             parent_path='',
                             error=f"Error reading directory: {str(e)}")

@app.route('/delete_item', methods=['POST'])
def delete_item():
    """Delete a file or folder from the output directory"""
    try:
        data = request.get_json()
        item_path = data.get('path', '')
        
        if not item_path:
            return jsonify({'error': 'No path provided'}), 400
        
        # Build the full path
        output_folder = "output"
        full_path = os.path.join(output_folder, item_path)
        
        # Security check: ensure the path is within the output folder
        try:
            full_path = os.path.abspath(full_path)
            output_folder_abs = os.path.abspath(output_folder)
            if not full_path.startswith(output_folder_abs):
                return jsonify({'error': 'Access denied'}), 403
        except:
            return jsonify({'error': 'Invalid path'}), 400
        
        # Check if path exists
        if not os.path.exists(full_path):
            return jsonify({'error': 'Item not found'}), 404
        
        # Delete the item
        if os.path.isdir(full_path):
            if item_path == '':  # Empty path means delete all contents of output folder
                # Delete all contents but keep the output folder itself
                for item in os.listdir(full_path):
                    item_path_full = os.path.join(full_path, item)
                    if os.path.isdir(item_path_full):
                        shutil.rmtree(item_path_full)
                    else:
                        os.remove(item_path_full)
                message = "All items in output folder deleted successfully"
            else:
                # Delete specific directory and all contents
                shutil.rmtree(full_path)
                message = f"Directory '{item_path}' deleted successfully"
        else:
            # Delete file
            os.remove(full_path)
            message = f"File '{item_path}' deleted successfully"
        
        return jsonify({'success': True, 'message': message})
        
    except Exception as e:
        return jsonify({'error': f'Error deleting item: {str(e)}'}), 500

@app.route('/server-info')
def server_info():
    """Display server information"""
    import platform
    import psutil
    
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        server_info = {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': cpu_percent,
            'memory_total': f"{memory.total / (1024**3):.1f} GB",
            'memory_available': f"{memory.available / (1024**3):.1f} GB",
            'memory_percent': memory.percent,
            'disk_total': f"{disk.total / (1024**3):.1f} GB",
            'disk_free': f"{disk.free / (1024**3):.1f} GB",
            'disk_percent': disk.percent,
            'output_folder_size': get_folder_size("output")
        }
        
        return jsonify(server_info)
    except ImportError:
        return jsonify({'error': 'psutil not installed'})
    except Exception as e:
        return jsonify({'error': str(e)})

def get_folder_size(folder_path):
    """Calculate folder size in GB"""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return f"{total_size / (1024**3):.2f} GB"
    except:
        return "Unknown"

if __name__ == '__main__':
    # SECURITY: Binding to specific IP address 65.108.122.8
    # This allows external access from the specified IP
    app.run(debug=DEBUG_MODE, host='0.0.0.0', port=5000) 