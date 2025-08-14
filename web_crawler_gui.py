"""
DeepSeek AI Web Crawler
Copyright (c) 2025 Ayaz Mensyoğlu

This file is part of the DeepSeek AI Web Crawler project.
Licensed under the Apache License, Version 2.0.
See NOTICE file for additional terms and conditions.
"""


import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import asyncio
import csv
import os
import json
from datetime import datetime
import queue
import sys
import traceback
import re

# Import the crawler modules
from main import crawl_from_sites_csv, read_sites_from_csv
from utils.scraper_utils import get_browser_config, get_llm_strategy, get_regex_strategy
from config import REQUIRED_KEYS, DEFAULT_CONFIG, ENV_VARS

# Add to top of web_crawler_gui.py
import sys
import os

def resource_path(relative_path):
    """Get absolute path to resource for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class EnhancedLogger:
    """Custom logger that filters and formats messages for the GUI"""
    
    def __init__(self, log_queue):
        self.log_queue = log_queue
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def write(self, text):
        if text.strip():
            # Filter and format the message
            formatted_message = self.format_message(text.strip())
            if formatted_message:
                self.log_queue.put(formatted_message)
    
    def flush(self):
        pass
    
    def format_message(self, message):
        """Format and filter messages for better readability"""
        
        # Crawling progress messages
        if "Loaded" in message and "sites to crawl" in message:
            match = re.search(r'Loaded (\d+) sites to crawl', message)
            if match:
                return f"📊 Loaded {match.group(1)} sites to crawl"
        
        # Site progress
        elif "Crawling site" in message:
            match = re.search(r'Crawling site (\d+)/(\d+)', message)
            if match:
                return f"🌐 Processing site {match.group(1)} of {match.group(2)}"
        
        # URL crawling
        elif "🔄 Crawling URL:" in message:
            url = message.split("🔄 Crawling URL: ")[1]
            domain = self.extract_domain(url)
            return f"🔍 Crawling: {domain}"
        
        # Products extracted
        elif "Extracted" in message and "venues from page" in message:
            match = re.search(r'Extracted (\d+) venues from page (\d+)', message)
            if match:
                return f"✅ Extracted {match.group(1)} products from page {match.group(2)}"
        
        # JS-based extraction
        elif "Extracted" in message and "products from page" in message:
            match = re.search(r'Extracted (\d+) products from page (\d+)', message)
            if match:
                return f"✅ Extracted {match.group(1)} products from page {match.group(2)}"
        
        # Added products
        elif "Added" in message and "unique products" in message:
            match = re.search(r'Added (\d+) unique products', message)
            if match:
                return f"📦 Added {match.group(1)} new products"
        
        # Pagination stopping
        elif "Stopping pagination" in message:
            return "🏁 Finished processing this site"
        
        # PDF processing messages
        elif "Found" in message and "PDF(s) for product:" in message:
            match = re.search(r'Found (\d+) PDF\(s\) for product: (.+)', message)
            if match:
                product_name = match.group(2)[:50] + "..." if len(match.group(2)) > 50 else match.group(2)
                return f"{'='*50}\n📄 Found {match.group(1)} PDF(s) for: {product_name}"
        
        elif "Downloaded PDF:" in message:
            pdf_path = message.split("Downloaded PDF: ")[1]
            filename = os.path.basename(pdf_path)
            return f"💾 Downloaded: {filename}"
        
        elif "Converting" in message and "to images" in message:
            pdf_path = message.split("Converting ")[1].split(" to images")[0]
            filename = os.path.basename(pdf_path)
            return f"🔄 Processing PDF: {filename}"
        
        elif "Cleaned PDF saved to:" in message:
            pdf_path = message.split("Cleaned PDF saved to: ")[1]
            filename = os.path.basename(pdf_path)
            return f"✨ Cleaned PDF saved: {filename}"
        
        # Error messages (keep important ones)
        elif "❌" in message or "⚠️" in message:
            return message
        
        # Skip verbose/debug messages
        elif any(skip in message for skip in [
            "Processing venue:", "Processing product:", "Duplicate venue", 
            "Duplicate:", "No products found", "No complete venues",
            "No content extracted", "processing page:", "Added",
            "QR detected", "Removed text", "Conversion complete"
        ]):
            return None
        
        # Skip other verbose messages
        return None
    
    def extract_domain(self, url):
        """Extract domain from URL for cleaner display"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return url[:30] + "..." if len(url) > 30 else url

class WebCrawlerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 DeepSeek AI Web Crawler")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Crawler state
        self.crawler_running = False
        self.crawler_thread = None
        self.log_queue = queue.Queue()
        
        # Progress tracking
        self.total_sites = 0
        self.current_site = 0
        self.total_products = 0
        self.current_products = 0
        
        # Configuration
        self.config = DEFAULT_CONFIG.copy()
        
        self.setup_ui()
        self.setup_logging()
        
        # Load API key from environment if available
        self.load_api_key_from_env()
        
        # Load user preferences
        self.load_user_preferences()
        
    def setup_ui(self):
        """Setup the main UI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🕷️DeepSeek AI Web Crawler", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File Selection Section
        self.setup_file_section(main_frame)
        
        # API Configuration Section
        self.setup_api_section(main_frame)
        
        # PDF Settings Section
        self.setup_pdf_section(main_frame)
        
        # Control Buttons Section
        self.setup_control_section(main_frame)
        
        # Progress Section
        self.setup_progress_section(main_frame)
        
        # Log Section
        self.setup_log_section(main_frame)
        
    def setup_file_section(self, parent):
        """Setup file selection section"""
        # File selection frame
        file_frame = ttk.LabelFrame(parent, text="CSV File Selection", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # CSV file selection
        ttk.Label(file_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.csv_path_var = tk.StringVar()
        csv_entry = ttk.Entry(file_frame, textvariable=self.csv_path_var, width=50)
        csv_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_csv_file)
        browse_btn.grid(row=0, column=2)
        
        # File info
        self.file_info_label = ttk.Label(file_frame, text="No file selected", foreground="gray")
        self.file_info_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
    def setup_api_section(self, parent):
        """Setup API configuration section"""
        # API configuration frame
        api_frame = ttk.LabelFrame(parent, text="API Configuration", padding="10")
        api_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        api_frame.columnconfigure(1, weight=1)
        
        # API Key
        ttk.Label(api_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.api_key_var = tk.StringVar()
        api_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=50, show="*")
        api_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        show_btn = ttk.Button(api_frame, text="Show/Hide", command=self.toggle_api_key_visibility)
        show_btn.grid(row=0, column=2)
        
        # Model Selection
        ttk.Label(api_frame, text="LLM Model:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        
        self.model_var = tk.StringVar(value=DEFAULT_CONFIG["default_model"])
        model_combo = ttk.Combobox(api_frame, textvariable=self.model_var, width=47, state="readonly")
        model_combo['values'] = DEFAULT_CONFIG["available_models"]
        model_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(5, 0))
        
        # API Key validation status
        self.api_status_label = ttk.Label(api_frame, text="⚠️ API key required", foreground="orange")
        self.api_status_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Test API Connection button
        test_api_btn = ttk.Button(api_frame, text="Test Connection", command=self.test_api_connection)
        test_api_btn.grid(row=2, column=2, pady=(5, 0))
        
        # Test Dependencies button
        test_deps_btn = ttk.Button(api_frame, text="Test Dependencies", command=self.test_dependencies)
        test_deps_btn.grid(row=2, column=3, pady=(5, 0), padx=(5, 0))
        
        # Environment file info
        env_info = ttk.Label(api_frame, text="Note: You can also set API key in .env file as GROQ_API_KEY", foreground="gray")
        env_info.grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))
        
        # Bind API key changes to validation
        self.api_key_var.trace('w', self.validate_api_key)
        
    def setup_pdf_section(self, parent):
        """Setup PDF download settings section"""
        # PDF settings frame
        pdf_frame = ttk.LabelFrame(parent, text="PDF Download Settings", padding="10")
        pdf_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        pdf_frame.columnconfigure(1, weight=1)
        
        # PDF Size Limit
        ttk.Label(pdf_frame, text="Max PDF Size (MB):").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.pdf_size_limit_var = tk.StringVar(value=str(DEFAULT_CONFIG["pdf_settings"]["max_file_size_mb"]))
        pdf_size_entry = ttk.Entry(pdf_frame, textvariable=self.pdf_size_limit_var, width=10)
        pdf_size_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        # Skip Large Files Checkbox
        self.skip_large_files_var = tk.BooleanVar(value=DEFAULT_CONFIG["pdf_settings"]["skip_large_files"])
        skip_large_check = ttk.Checkbutton(pdf_frame, text="Skip files larger than size limit", 
                                         variable=self.skip_large_files_var)
        skip_large_check.grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        
        # Help text
        help_text = ttk.Label(pdf_frame, text="PDFs larger than this size will be skipped during download", 
                             foreground="gray")
        help_text.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Bind PDF settings changes to save preferences
        self.pdf_size_limit_var.trace('w', lambda *args: self.save_user_preferences())
        self.skip_large_files_var.trace('w', lambda *args: self.save_user_preferences())
        
    def setup_control_section(self, parent):
        """Setup control buttons section"""
        # Control frame
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=4, column=0, columnspan=3, pady=(0, 10))
        
        # Start button
        self.start_btn = ttk.Button(control_frame, text="Start Crawling", 
                                   command=self.start_crawling, style="Accent.TButton")
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Stop button
        self.stop_btn = ttk.Button(control_frame, text="Stop Crawling", 
                                  command=self.stop_crawling, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        
    def setup_progress_section(self, parent):
        """Setup progress tracking section"""
        # Progress frame
        progress_frame = ttk.LabelFrame(parent, text="Crawling Progress", padding="10")
        progress_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Progress labels
        progress_info_frame = ttk.Frame(progress_frame)
        progress_info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        progress_info_frame.columnconfigure(1, weight=1)
        
        ttk.Label(progress_info_frame, text="Current Site:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.current_site_label = ttk.Label(progress_info_frame, text="Not started", foreground="gray")
        self.current_site_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(progress_info_frame, text="Current Page:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.current_page_label = ttk.Label(progress_info_frame, text="Not started", foreground="gray")
        self.current_page_label.grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(progress_info_frame, text="Products Found:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10))
        self.products_found_label = ttk.Label(progress_info_frame, text="0", foreground="gray")
        self.products_found_label.grid(row=2, column=1, sticky=tk.W)
        
        # Enhanced progress info
        ttk.Label(progress_info_frame, text="PDFs Processed:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10))
        self.pdfs_processed_label = ttk.Label(progress_info_frame, text="0", foreground="gray")
        self.pdfs_processed_label.grid(row=3, column=1, sticky=tk.W)
        
    def setup_log_section(self, parent):
        """Setup log display section"""
        # Log frame
        log_frame = ttk.LabelFrame(parent, text="Crawling Logs", padding="10")
        log_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(6, weight=1)
        
        # Log text area with better styling
        self.log_text = scrolledtext.ScrolledText(
            log_frame, 
            height=15, 
            width=80, 
            font=('Consolas', 9),
            bg='#f8f9fa',
            fg='#212529',
            insertbackground='#212529'
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Log controls
        log_controls_frame = ttk.Frame(log_frame)
        log_controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        clear_log_btn = ttk.Button(log_controls_frame, text="Clear Logs", 
                                  command=self.clear_logs)
        clear_log_btn.pack(side=tk.LEFT)
        
        save_log_btn = ttk.Button(log_controls_frame, text="Save Logs", 
                                 command=self.save_logs)
        save_log_btn.pack(side=tk.LEFT, padx=(10, 0))
        
    def setup_logging(self):
        """Setup enhanced logging to capture print statements"""
        self.enhanced_logger = EnhancedLogger(self.log_queue)
        sys.stdout = self.enhanced_logger
        sys.stderr = self.enhanced_logger
        
        # Start log processing
        self.process_log_queue()
        
    def process_log_queue(self):
        """Process log messages from the queue"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.add_log_message(message)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_log_queue)
    
    def add_log_message(self, message):
        """Add a formatted message to the log with timestamp and color coding"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Determine color based on message type
        if "✅" in message or "✨" in message:
            color = "#28a745"  # Green for success
        elif "❌" in message or "⚠️" in message:
            color = "#dc3545"  # Red for errors/warnings
        elif "📊" in message or "🌐" in message:
            color = "#007bff"  # Blue for info
        elif "📄" in message or "💾" in message:
            color = "#6f42c1"  # Purple for PDF operations
        elif "🔄" in message or "🔍" in message:
            color = "#fd7e14"  # Orange for processing
        else:
            color = "#212529"  # Default dark gray
        
        # Insert message with color
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        
        # Apply color to the last line
        last_line_start = self.log_text.index("end-2c linestart")
        last_line_end = self.log_text.index("end-1c")
        self.log_text.tag_add(f"color_{color}", last_line_start, last_line_end)
        self.log_text.tag_config(f"color_{color}", foreground=color)
        
        # Auto-scroll to bottom
        self.log_text.see(tk.END)
        self.log_text.update_idletasks()
        
        # Update progress based on message content
        self.update_progress_from_message(message)
            
    def update_progress_from_message(self, message):
        """Update progress indicators based on log messages"""
        # Update site progress
        if "Processing site" in message:
            match = re.search(r'Processing site (\d+) of (\d+)', message)
            if match:
                self.current_site = int(match.group(1))
                self.total_sites = int(match.group(2))
                self.current_site_label.config(text=f"{self.current_site}/{self.total_sites}")
                
                # Update progress bar
                if self.total_sites > 0:
                    progress = (self.current_site - 1) / self.total_sites * 100
                    self.progress_var.set(progress)
        
        # Update products count
        elif "Added" in message and "new products" in message:
            match = re.search(r'Added (\d+) new products', message)
            if match:
                self.current_products += int(match.group(1))
                self.products_found_label.config(text=str(self.current_products))
        
        # Update PDFs processed
        elif "Cleaned PDF saved:" in message:
            current_pdfs = int(self.pdfs_processed_label.cget("text"))
            self.pdfs_processed_label.config(text=str(current_pdfs + 1))
            
    def browse_csv_file(self):
        """Browse for CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.csv_path_var.set(filename)
            self.update_file_info(filename)
            
    def update_file_info(self, filename):
        """Update file information display"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                sites = list(reader)
                self.file_info_label.config(
                    text=f"Loaded {len(sites)} sites from {os.path.basename(filename)}",
                    foreground="green"
                )
        except Exception as e:
            self.file_info_label.config(
                text=f"Error reading file: {str(e)}",
                foreground="red"
            )
            
    def toggle_api_key_visibility(self):
        """Toggle API key visibility"""
        # Find the API key entry widget in the API frame
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.LabelFrame) and child.cget('text') == "API Configuration":
                        for grandchild in child.winfo_children():
                            if isinstance(grandchild, ttk.Entry) and grandchild.cget('show') in ['*', '']:
                                current_show = grandchild.cget('show')
                                grandchild.config(show='' if current_show == '*' else '*')
                                return
            
    def validate_api_key(self, *args):
        """Validate the API key and update the status label"""
        api_key = self.api_key_var.get()
        if api_key:
            self.api_status_label.config(text="✅ API key valid", foreground="green")
        else:
            self.api_status_label.config(text="⚠️ API key required", foreground="orange")
            
    def validate_inputs(self):
        """Validate user inputs before starting crawling"""
        if not self.csv_path_var.get():
            messagebox.showerror("Error", "Please select a CSV file")
            return False
            
        if not os.path.exists(self.csv_path_var.get()):
            messagebox.showerror("Error", "Selected CSV file does not exist")
            return False
            
        # Check CSV format
        try:
            with open(self.csv_path_var.get(), 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                required_columns = ['url', 'css_selector', 'button_selector']
                if not all(col in reader.fieldnames for col in required_columns):
                    messagebox.showerror("Error", 
                                       f"CSV must contain columns: {', '.join(required_columns)}")
                    return False
        except Exception as e:
            messagebox.showerror("Error", f"Error reading CSV file: {str(e)}")
            return False
            
        # Validate API key
        if not self.api_key_var.get():
            messagebox.showerror("Error", "API Key is required. Please enter it in the API Configuration section.")
            return False
            
        return True
        
    def start_crawling(self):
        """Start the crawling process in a separate thread"""
        if not self.validate_inputs():
            return
            
        if self.crawler_running:
            messagebox.showwarning("Warning", "Crawler is already running")
            return
            
        # Set API key in environment if provided
        if self.api_key_var.get():
            os.environ['GROQ_API_KEY'] = self.api_key_var.get()
            
        # Set PDF size limit
        pdf_size_limit = int(self.pdf_size_limit_var.get())
        skip_large_files = self.skip_large_files_var.get()
        if skip_large_files:
            from utils.scraper_utils import set_pdf_size_limit
            set_pdf_size_limit(pdf_size_limit)
            self.add_log_message(f"📏 PDF size limit set to {pdf_size_limit}MB")
            
        # Update UI state
        self.crawler_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Reset progress
        self.progress_var.set(0)
        self.current_site_label.config(text="Starting...")
        self.current_page_label.config(text="Starting...")
        self.products_found_label.config(text="0")
        self.pdfs_processed_label.config(text="0")
        
        # Reset counters
        self.current_products = 0
        self.current_site = 0
        self.total_sites = 0
        
        # Start crawling in separate thread
        self.crawler_thread = threading.Thread(target=self.run_crawler, daemon=True)
        self.crawler_thread.start()
        
        self.add_log_message("🚀 Starting web crawler...")
        
    def run_crawler(self):
        """Run the crawler in a separate thread"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the crawler with API key and model
            loop.run_until_complete(self.crawl_with_progress(
                self.csv_path_var.get(),
                self.api_key_var.get(),
                self.model_var.get()
            ))
            
        except Exception as e:
            self.add_log_message(f"❌ Error during crawling: {str(e)}")
            self.add_log_message(traceback.format_exc())
        finally:
            # Update UI on main thread
            self.root.after(0, self.crawling_finished)
            
    async def crawl_with_progress(self, csv_path, api_key, model):
        """Run crawler with progress updates"""
        try:
            # Read sites
            sites = read_sites_from_csv(csv_path)
            total_sites = len(sites)
            
            self.add_log_message(f"📊 Loaded {total_sites} sites to crawl")
            
            # Create output directory
            os.makedirs(self.config['output_folder'], exist_ok=True)
            
            # Run the crawler
            await crawl_from_sites_csv(csv_path, api_key, model)
            
        except Exception as e:
            self.add_log_message(f"❌ Crawling error: {str(e)}")
            raise
            
    def stop_crawling(self):
        """Stop the crawling process"""
        if not self.crawler_running:
            return
            
        self.crawler_running = False
        self.add_log_message("🛑 Stopping crawler...")
        
        # Update UI
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.current_site_label.config(text="Stopped")
        self.current_page_label.config(text="Stopped")
        
    def crawling_finished(self):
        """Called when crawling finishes"""
        self.crawler_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        self.add_log_message("✅ Crawling completed!")
        self.progress_var.set(100)
        
    def clear_logs(self):
        """Clear the log display"""
        self.log_text.delete(1.0, tk.END)
        
    def save_logs(self):
        """Save logs to a file"""
        filename = filedialog.asksaveasfilename(
            title="Save Logs",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Logs saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving logs: {str(e)}")
                
    def load_api_key_from_env(self):
        """Load API key from environment variables if available"""
        # Check for different API providers
        api_key = None
        provider = None
        
        for provider_name, env_var in ENV_VARS.items():
            api_key = os.environ.get(env_var)
            if api_key:
                provider = provider_name
                break
        
        if api_key:
            self.api_key_var.set(api_key)
            self.validate_api_key() # Re-validate to show green status
            self.add_log_message(f"✅ API key loaded from environment variable {env_var}")
            
            # Auto-select appropriate model based on provider
            if provider == "GROQ_API_KEY":
                if self.model_var.get() not in ["groq/deepseek-r1-distill-llama-70b", "groq/llama3-8b-8192", "groq/llama3-70b-8192", "groq/mixtral-8x7b-32768"]:
                    self.model_var.set("groq/deepseek-r1-distill-llama-70b")
            elif provider == "OPENAI_API_KEY":
                if self.model_var.get() not in ["openai/gpt-4o", "openai/gpt-4o-mini"]:
                    self.model_var.set("openai/gpt-4o")
            elif provider == "ANTHROPIC_API_KEY":
                if self.model_var.get() not in ["anthropic/claude-3-5-sonnet-20241022", "anthropic/claude-3-haiku-20240307"]:
                    self.model_var.set("anthropic/claude-3-5-sonnet-20241022")
        else:
            self.add_log_message("⚠️ API key not found in environment variables. Please enter it in the GUI.")

    def save_user_preferences(self):
        """Save user preferences to a JSON file"""
        try:
            preferences = {
                "api_key": self.api_key_var.get(),
                "model": self.model_var.get(),
                "csv_file": self.csv_path_var.get(),
                "output_folder": self.config.get("output_folder", "output"),
                "pdf_size_limit": self.pdf_size_limit_var.get(),
                "skip_large_files": self.skip_large_files_var.get()
            }
            
            with open("user_preferences.json", "w") as f:
                json.dump(preferences, f, indent=2)
                
        except Exception as e:
            self.add_log_message(f"⚠️ Could not save preferences: {str(e)}")

    def load_user_preferences(self):
        """Load user preferences from JSON file"""
        try:
            if os.path.exists("user_preferences.json"):
                with open("user_preferences.json", "r") as f:
                    preferences = json.load(f)
                
                # Only load preferences if they're not already set by environment
                if not self.api_key_var.get() and preferences.get("api_key"):
                    self.api_key_var.set(preferences["api_key"])
                
                if preferences.get("model"):
                    self.model_var.set(preferences["model"])
                
                if preferences.get("csv_file"):
                    self.csv_path_var.set(preferences["csv_file"])
                    self.update_file_info(preferences["csv_file"])
                
                if preferences.get("output_folder"):
                    self.config["output_folder"] = preferences["output_folder"]
                
                if preferences.get("pdf_size_limit"):
                    self.pdf_size_limit_var.set(preferences["pdf_size_limit"])
                
                if preferences.get("skip_large_files") is not None:
                    self.skip_large_files_var.set(preferences["skip_large_files"])
                    
                self.add_log_message("✅ User preferences loaded")
                
        except Exception as e:
            self.add_log_message(f"⚠️ Could not load preferences: {str(e)}")

    def test_api_connection(self):
        """Test the API connection with the current API key and model"""
        api_key = self.api_key_var.get()
        model = self.model_var.get()

        if not api_key:
            messagebox.showwarning("Warning", "Please enter an API key in the API Configuration section.")
            return

        self.add_log_message(f"🔍 Testing API connection with model: {model}")

        # Run test in a separate thread
        test_thread = threading.Thread(target=self._run_api_test, args=(api_key, model), daemon=True)
        test_thread.start()

    def _run_api_test(self, api_key, model):
        """Run the API test in a separate thread"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the test
            result = loop.run_until_complete(self._test_api_async(api_key, model))
            
            # Update UI on main thread
            if result:
                self.root.after(0, lambda: self.add_log_message("✅ API connection test successful!"))
                self.root.after(0, lambda: messagebox.showinfo("Success", "API connection test successful!"))
            else:
                self.root.after(0, lambda: self.add_log_message("❌ API connection test failed"))
                self.root.after(0, lambda: messagebox.showerror("Error", "API connection test failed. Please check your API key and model."))
                
        except Exception as e:
            self.root.after(0, lambda: self.add_log_message(f"❌ API test error: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"API test error: {str(e)}"))

    async def _test_api_async(self, api_key, model):
        """Async method to test API connection"""
        try:
            from utils.scraper_utils import get_llm_strategy
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
            
            # Create LLM strategy with the provided API key and model
            llm_strategy = get_llm_strategy(api_key=api_key, model=model)
            
            # Create a simple test crawler
            from utils.scraper_utils import get_browser_config
            browser_config = get_browser_config()
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                # Test with a simple HTML snippet
                test_html = """
                <div class="product">
                    <h3>Test Product</h3>
                    <a href="https://example.com/product/1">View Product</a>
                </div>
                """
                
                result = await crawler.arun(
                    url=f"raw:\nhttps://example.com\n{test_html}",
                    config=CrawlerRunConfig(
                        extraction_strategy=llm_strategy,
                        cache_mode=CacheMode.BYPASS,
                        session_id="api_test_session"
                    )
                )
                
                if result.success and result.extracted_content:
                    self.add_log_message("✅ API test completed successfully!")
                    return True
                else:
                    self.add_log_message(f"❌ API test failed: {result.error_message}")
                    return False
                    
        except Exception as e:
            self.add_log_message(f"❌ API test exception: {str(e)}")
            return False

    def test_dependencies(self):
        """Test Tesseract and Poppler installations."""
        self.add_log_message("🔍 Testing Tesseract and Poppler...")
        try:
            # Test Tesseract
            self.add_log_message("   - Checking Tesseract installation...")
            import pytesseract
            try:
                pytesseract.get_languages()
                self.add_log_message("    ✅ Tesseract is installed and working.")
            except ImportError:
                self.add_log_message("    ❌ Tesseract is not installed. Please install Tesseract OCR.")
                self.add_log_message("     - On Windows: `choco install tesseract-ocr` or `winget install tesseract-ocr`")
                self.add_log_message("     - On macOS: `brew install tesseract`")
                self.add_log_message("     - On Linux: `sudo apt-get install tesseract-ocr` or `sudo apt-get install libtesseract-dev`")
                self.add_log_message("     - For more details, see https://github.com/madmaze/pytesseract/wiki/Installation")
            except Exception as e:
                self.add_log_message(f"    ❌ Tesseract test failed: {e}")

            # Test Poppler
            self.add_log_message("   - Checking Poppler installation...")
            import fitz # PyMuPDF
            try:
                # Try to open a dummy PDF to check if Poppler is installed
                fitz.open("dummy.pdf")
                self.add_log_message("    ✅ Poppler is installed and working.")
            except ImportError:
                self.add_log_message("    ❌ Poppler is not installed. Please install Poppler.")
                self.add_log_message("     - On Windows: `choco install poppler` or `winget install poppler`")
                self.add_log_message("     - On macOS: `brew install poppler`")
                self.add_log_message("     - On Linux: `sudo apt-get install poppler-utils`")
                self.add_log_message("     - For more details, see https://github.com/pymupdf/PyMuPDF/wiki/Installation")
            except Exception as e:
                self.add_log_message(f"    ❌ Poppler test failed: {e}")

            self.add_log_message("✅ Dependency tests completed.")
            messagebox.showinfo("Dependency Tests", "Dependency tests completed. Please check the logs for details.")

        except Exception as e:
            self.add_log_message(f"❌ Error during dependency tests: {str(e)}")
            messagebox.showerror("Dependency Tests Error", f"Error during dependency tests: {str(e)}")

    def on_closing(self):
        """Handle application closing"""
        # Save user preferences before closing
        self.save_user_preferences()
        
        if self.crawler_running:
            if messagebox.askokcancel("Quit", "Crawler is running. Do you want to quit?"):
                self.stop_crawling()
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = WebCrawlerGUI(root)
    
    # Set closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Configure style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Run the application
    root.mainloop()

if __name__ == "__main__":
    main() 