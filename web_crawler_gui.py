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
from io import StringIO
import traceback
import re

# Import the crawler modules
from main import crawl_from_sites_csv, read_sites_from_csv
from utils.scraper_utils import get_browser_config, get_llm_strategy, get_regex_strategy
from config import REQUIRED_KEYS

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
                return f"📄 Found {match.group(1)} PDF(s) for: {product_name}"
        
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
        self.config = {
            'api_key': '',
            'csv_file': '',
            'output_folder': 'output'
        }
        
        self.setup_ui()
        self.setup_logging()
        
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
        
        # Environment file info
        env_info = ttk.Label(api_frame, text="Note: You can also set API key in .env file", foreground="gray")
        env_info.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
    def setup_control_section(self, parent):
        """Setup control buttons section"""
        # Control frame
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=3, column=0, columnspan=3, pady=(0, 10))
        
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
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
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
        log_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(5, weight=1)
        
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
        current_show = self.api_key_var.get()
        if current_show:
            # Find the entry widget and change its show attribute
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.LabelFrame):
                            for grandchild in child.winfo_children():
                                if isinstance(grandchild, ttk.Entry):
                                    if grandchild.cget('show') == '*':
                                        grandchild.config(show='')
                                    else:
                                        grandchild.config(show='*')
                                    break
                                    
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
            
            # Run the crawler
            loop.run_until_complete(self.crawl_with_progress())
            
        except Exception as e:
            self.add_log_message(f"❌ Error during crawling: {str(e)}")
            self.add_log_message(traceback.format_exc())
        finally:
            # Update UI on main thread
            self.root.after(0, self.crawling_finished)
            
    async def crawl_with_progress(self):
        """Run crawler with progress updates"""
        try:
            # Read sites
            sites = read_sites_from_csv(self.csv_path_var.get())
            total_sites = len(sites)
            
            self.add_log_message(f"📊 Loaded {total_sites} sites to crawl")
            
            # Get configurations
            browser_config = get_browser_config()
            llm_strategy = get_llm_strategy()
            regex_strategy = get_regex_strategy()
            
            # Create output directory
            os.makedirs(self.config['output_folder'], exist_ok=True)
            
            # Run the crawler
            await crawl_from_sites_csv(self.csv_path_var.get())
            
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
                
    def on_closing(self):
        """Handle application closing"""
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