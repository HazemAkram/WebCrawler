"""
DeepSeek AI Web Crawler
Copyright (c) 2025 Ayaz Mensyoğlu

This file is part of the DeepSeek AI Web Crawler project.
Licensed under the Apache License, Version 2.0.
See NOTICE file for additional terms and conditions.
"""


import json
import os
import hashlib
import traceback
import asyncio
import gc 
import shutil  # Add import for file copying operations

import aiofiles
import aiohttp

from typing import List, Set, Tuple
from fake_useragent import UserAgent

from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import re

# AI-powered PDF processing is now handled automatically in cleaner.py
from cleaner import pdf_processing
from dotenv import load_dotenv

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    LLMConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMExtractionStrategy,
    RegexExtractionStrategy,
)

from models.venue import Venue, PDF
from utils.data_utils import is_complete_venue, is_duplicate_venue
from config import DEFAULT_CONFIG



load_dotenv()

log_callback = None

def set_log_callback(callback):
    """Set the logging callback function for web interface integration"""
    global log_callback
    log_callback = callback

# Get PDF size limit from configuration
MAX_SIZE_MB = DEFAULT_CONFIG.get("pdf_settings", {}).get("max_file_size_mb", 10)

def set_pdf_size_limit(size_mb: int):
    """
    Set the maximum PDF file size limit for downloads.
    
    Args:
        size_mb (int): Maximum file size in megabytes
    """
    global MAX_SIZE_MB
    MAX_SIZE_MB = size_mb
    log_message(f"PDF size limit set to {MAX_SIZE_MB}MB", "INFO")

def get_pdf_size_limit() -> int:
    """
    Get the current maximum PDF file size limit for downloads.
    
    Returns:
        int: Current maximum file size limit in megabytes
    """
    return MAX_SIZE_MB

def log_message(message, level="INFO"):
    """Log a message, either to console or web interface"""
    global log_callback
    if log_callback:
        log_callback(message, level)
    else:
        print(f"[{level}] {message}")


def sanitize_folder_name(product_name: str) -> str:
    """
    Sanitizes a product name to be used as a folder name.
    Removes or replaces invalid characters that could cause folder creation issues.
    
    Args:
        product_name (str): The original product name
        
    Returns:
        str: Sanitized folder name safe for filesystem use
    """
    # Remove or replace invalid characters for folder names
    # Windows: < > : " | ? * \ /
    # Unix/Linux: / (forward slash)
    # Common problematic characters: \ / : * ? " < > |
    
    # Replace backslashes and forward slashes with underscores
    sanitized = product_name.replace('\\', '_').replace('/', '_')
    
    # Replace other invalid characters
    invalid_chars = r'[<>:"|?*]'
    sanitized = re.sub(invalid_chars, '_', sanitized)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Replace multiple consecutive underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Limit length to prevent filesystem issues (max 255 characters)
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    # Ensure the name is not empty
    if not sanitized:
        sanitized = "unnamed_product"
    
    return sanitized


def generate_pdf_filename_from_llm_text(pdf_text: str, pdf_type: str, pdf_language: str, product_name: str) -> str:
    """
    Generate an appropriate filename for a PDF using LLM extracted text.
    
    Args:
        pdf_text (str): The text extracted by LLM from the PDF link
        pdf_type (str): The type of PDF (Data Sheet, Technical Drawing, etc.)
        pdf_language (str): The language of the PDF
        product_name (str): The name of the product for fallback naming
        
    Returns:
        str: A sanitized filename with .pdf extension
    """
    
    # Clean and prepare the text for filename
    if pdf_text and pdf_text.strip():
        # Use the LLM extracted text as base filename
        base_filename = pdf_text.strip()
        
        # Remove common unwanted prefixes/suffixes
        unwanted_patterns = [
            r'^download\s*',
            r'^pdf\s*',
            r'^document\s*',
            r'\s*download$',
            r'\s*pdf$',
            r'\s*document$',
            r'^\d+\.\s*',  # Remove leading numbers like "1. "
            r'^\d+\s*',    # Remove leading numbers
        ]
        
        for pattern in unwanted_patterns:
            base_filename = re.sub(pattern, '', base_filename, flags=re.IGNORECASE)
        
        # Clean up any extra whitespace
        base_filename = base_filename.strip()
        
        # If the text becomes empty after cleaning, fall back to product name
        if not base_filename:
            base_filename = sanitize_folder_name(product_name)
        
        # Add type and language info for better identification
        type_suffix = ""
        if pdf_type and pdf_type.lower() not in ['unknown', '']:
            type_suffix += f"_{pdf_type.replace(' ', '_')}"
        
        if pdf_language and pdf_language.lower() not in ['unknown', 'en', 'english']:
            type_suffix += f"_{pdf_language.upper()}"
        
        # Combine base filename with type info
        if type_suffix:
            filename = f"{base_filename}{type_suffix}.pdf"
        else:
            filename = f"{base_filename}.pdf"
    else:
        # Fallback to product name if no text available
        filename = f"{sanitize_folder_name(product_name)}.pdf"
    
    # Sanitize the filename
    filename = sanitize_filename(filename)
    
    return filename


def generate_pdf_filename(pdf_url: str, product_name: str) -> str:
    """
    Generate an appropriate filename for a PDF URL, handling extensionless URLs.
    This is kept for backward compatibility but should be replaced with LLM-based naming.
    
    Args:
        pdf_url (str): The URL of the PDF file
        product_name (str): The name of the product for fallback naming
        
    Returns:
        str: A sanitized filename with .pdf extension
    """
    
    # Parse the URL
    parsed_url = urlparse(pdf_url)
    path = parsed_url.path
    query = parsed_url.query
    
    # Try to extract filename from URL path
    if path and path != '/':
        # Get the last part of the path
        path_parts = [part for part in path.split('/') if part]
        if path_parts:
            filename = path_parts[-1]
            # Remove any existing extension
            if '.' in filename:
                filename = filename.rsplit('.', 1)[0]
            # Add .pdf extension
            filename = f"{filename}.pdf"
        else:
            # No meaningful path, use query parameters or fallback
            filename = generate_filename_from_query(query, product_name)
    else:
        # No path, use query parameters or fallback
        filename = generate_filename_from_query(query, product_name)
    
    # Sanitize the filename
    filename = sanitize_filename(filename)
    
    return filename


def generate_filename_from_query(query: str, product_name: str) -> str:
    """
    Generate filename from URL query parameters or product name.
    
    Args:
        query (str): URL query string
        product_name (str): Product name for fallback
        
    Returns:
        str: Generated filename
    """
    if query:
        # Try to extract meaningful parameters
        params = parse_qs(query)
        
        # Look for common ID parameters
        for param_name in ['id', 'file_id', 'doc_id', 'download_id']:
            if param_name in params:
                return f"{param_name}_{params[param_name][0]}.pdf"
        
        # Look for type parameters
        for param_name in ['type', 'file_type', 'doc_type']:
            if param_name in params:
                return f"document_{params[param_name][0]}.pdf"
        
        # Use first parameter as fallback
        first_param = list(params.keys())[0]
        first_value = params[first_param][0]
        return f"{first_param}_{first_value}.pdf"
    
    # Ultimate fallback: use product name
    return f"{sanitize_folder_name(product_name)}.pdf"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to be safe for filesystem use.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = r'[<>:"|?*\x00-\x1f]'
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Replace multiple consecutive underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Limit length to prevent filesystem issues
    if len(sanitized) > 200:  # Leave room for path
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:200-len(ext)-1] + (f'.{ext}' if ext else '')
    
    # Ensure the name is not empty
    if not sanitized:
        sanitized = "unnamed_document.pdf"
    
    return sanitized


# https://www.ors.com.tr/en/tek-sirali-sabit-bilyali-rulmanlar
async def download_pdf_links(
        crawler: AsyncWebCrawler, 
        product_url: str, 
        product_name: str,
        output_folder: str, 
        pdf_llm_strategy: LLMExtractionStrategy,
        pdf_selector: str,
        session_id="pdf_download_session", 
        regex_strategy: RegexExtractionStrategy = None,
        domain_name: str = None,
        api_key: str = None,
        cat_name: str = "Uncategorized",
        ):
    
    """
    Opens the given product page and uses CSS selector to target PDF links.
    Uses LLMExtractionStrategy to identify and extract technical PDF documents,
    then downloads them. Prevents downloading duplicate PDFs by checking existing files.
    Creates a category-based folder structure: output/{category}/{product}
    
    Args:
        crawler: The AsyncWebCrawler instance
        product_url: URL of the product page
        product_name: Name of the product for folder creation
        output_folder: Base output directory for downloads
        pdf_llm_strategy: LLM extraction strategy for PDF processing
        pdf_selector: CSS selector to target PDF link elements
        session_id: Session identifier for crawling
        regex_strategy: Optional regex extraction strategy (unused)
        domain_name: Domain name for URL completion
        api_key: API key for LLM provider
        cat_name: Category name from CSV for folder organization (defaults to "Uncategorized")
    """




    # Global dictionary to track downloaded PDFs with their file paths across all products
    if not hasattr(download_pdf_links, 'downloaded_pdfs'):
        download_pdf_links.downloaded_pdfs = {}  # Change from set to dict: {url: file_path}

    try:
        log_message(f"🔍 Starting PDF extraction for product: {product_name}", "INFO")
        log_message(f"📍 Using pdf selector: {pdf_selector}", "INFO")


        product_url = f"{product_url}"
        # Crawl the page with CSS selector targeting PDF links
        pdf_result = await crawler.arun(
            url=product_url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                extraction_strategy=pdf_llm_strategy,
                target_elements=pdf_selector,
                session_id=f"{session_id}_pdf_extraction",
                scan_full_page=True,
                remove_overlay_elements=True,
                verbose=True,
                simulate_user=True,
            )
        )

        if not pdf_result.success or not pdf_result.extracted_content:
            log_message(f"❌ Failed to extract PDF content for product: {product_name}", "ERROR")
            log_message(f"🔗 Product URL: {product_url}", "INFO")
            return

        # Step 2: Parse extracted content
        try:
            extracted_data = json.loads(pdf_result.extracted_content)
        except json.JSONDecodeError as e:
            log_message(f"❌ Failed to parse extracted JSON content: {e}", "ERROR")
            return

        if not extracted_data:
            log_message(f"📭 No PDFs found for product: {product_name}", "INFO")
            return

        log_message(f"🔗 Found {len(extracted_data)} potential PDF(s) via CSS selector", "INFO")

        # Step 3: Process each PDF link to download it
        pdf_links = []
        seen_pdf_urls_in_page = set()

        for item in extracted_data:
            # Stop if we already have 3 PDFs
            if len(pdf_links) >= 3:
                log_message(f"📊 Reached maximum limit of 3 PDFs, stopping validation", "INFO")
                break
            
            pdf_url = item.get('url', '')
            
            
            if not pdf_url:
                log_message(f"⚠️ Skipping item with missing URL: {item}", "WARNING")
                continue
            
            
            # Convert relative URLs to absolute
            if not (pdf_url.startswith("https://") or pdf_url.startswith("http://") or pdf_url.startswith("www")):
                if domain_name:
                    pdf_url = f"https://{domain_name}{pdf_url}"
                    item['url'] = pdf_url  # Update the item with the corrected URL
                else:
                    log_message(f"⚠️ Skipping relative URL without domain: {pdf_url}", "WARNING")
                    continue
            
            # Check for duplicates within this page
            if pdf_url not in seen_pdf_urls_in_page:
                pdf_links.append(item)
                seen_pdf_urls_in_page.add(pdf_url)
                log_message(f"✅ Validated PDF ({len(pdf_links)}/3): {item.get('type', 'Unknown')} - {item.get('text', 'Unknown')}", "INFO")
            else: 
                log_message(f"⏭️ Skipping duplicate PDF URL: {pdf_url}", "INFO")

        # Check if any PDFs were found
        if not pdf_links:
            log_message(f"📭 No PDFs found on page for product: {product_name}", "INFO")
            log_message(f"🔗 Product URL: {product_url}", "INFO")
            return  # Exit early without creating any folders

        log_message(f"📄 Found {len(pdf_links)} PDF(s) for product: {product_name}", "INFO")

        # Create the download folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Create category-based folder structure: output/{category}/{product}
        sanitized_cat_name = sanitize_folder_name(cat_name)
        category_path = os.path.join(output_folder, sanitized_cat_name)
        os.makedirs(category_path, exist_ok=True)
        
        productPath = os.path.join(category_path, sanitize_folder_name(product_name))
        if not os.path.exists(productPath):
            os.makedirs(productPath)
            log_message(f"📁 Created folder structure: {sanitized_cat_name}/{sanitize_folder_name(product_name)}", "INFO")


        # Enhanced sorting: Priority first, then by document type preference, then by language
        def sort_key(pdf):
            priority_score = {'High': 3, 'Medium': 2, 'Low': 1}.get(pdf.get('priority', 'Unknown'), 0)
            
            # Type preference: Data Sheet > Technical Drawing > Manual
            type_score = 0
            pdf_type = pdf.get('type', '').lower()
            if 'data sheet' in pdf_type or 'datasheet' in pdf_type or 'specification' in pdf_type:
                type_score = 3
            elif 'drawing' in pdf_type or 'dimensional' in pdf_type:
                type_score = 2
            elif 'manual' in pdf_type or 'guide' in pdf_type:
                type_score = 1
            
            # Language preference: English > German > Turkish > Others
            language_score = 0
            pdf_language = pdf.get('language', '').lower()
            if 'english' in pdf_language:
                language_score = 4
            elif 'german' in pdf_language or 'deutsch' in pdf_language:
                language_score = 3
            elif 'turkish' in pdf_language or 'türkçe' in pdf_language:
                language_score = 2
            else:
                language_score = 1
            
            return (priority_score, type_score, language_score)

        pdf_links.sort(key=sort_key, reverse=True)

        log_message(f"📊 Processing {len(pdf_links)} validated technical documents (max 3) by priority", "INFO")

        # Log selected document summary
        if pdf_links:
            log_message("📋 Selected documents:", "INFO")
            for i, pdf in enumerate(pdf_links, 1):
                log_message(f"   {i}. {pdf.get('type', 'Unknown')} ({pdf.get('language', 'Unknown')}) - Priority: {pdf.get('priority', 'Unknown')}", "INFO")

        # Download each PDF with duplicate checking (SSL verification disabled)
        log_message(f"📥 Starting download of {len(pdf_links)} PDF documents", "INFO")

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            for i, pdf_info in enumerate(pdf_links, 1):
                pdf_url = pdf_info['url']
                pdf_text = pdf_info['text']
                pdf_type = pdf_info['type']
                pdf_language = pdf_info.get('language', 'Unknown')
                pdf_priority = pdf_info.get('priority', 'Unknown')
                priority_emoji = "🔴" if pdf_priority == 'High' else "🟡" if pdf_priority == 'Medium' else "🟢"
                log_message(f"{priority_emoji} Downloading PDF {i}/{len(pdf_links)}", "INFO")
                # Generate filename using LLM extracted text
                filename = generate_pdf_filename_from_llm_text(pdf_text, pdf_type, pdf_language, product_name)
                save_path = os.path.join(productPath, filename)
                
                # # Check if this exact PDF URL has been downloaded before (global tracking)
                # if pdf_url in download_pdf_links.downloaded_pdfs:
                #     previous_path = download_pdf_links.downloaded_pdfs[pdf_url]
                #     if os.path.exists(previous_path):
                #         try:
                #             # Copy the previously cleaned PDF to current product folder with new filename
                #             shutil.copy2(previous_path, save_path)
                #             log_message(f"📋 Copied cleaned PDF from previous download: {filename}", "INFO")
                #             continue
                #         except Exception as copy_error:
                #             log_message(f"⚠️ Failed to copy cleaned PDF: {str(copy_error)}", "WARNING")
                #             log_message(f"   Will re-download and process: {filename}", "INFO")
                #     else:
                #         log_message(f"⚠️ Previously downloaded PDF not found at: {previous_path}", "WARNING")
                #         log_message(f"   Will re-download and process: {filename}", "INFO")
                
                # # Check if file already exists in the product folder
                # if os.path.exists(save_path):
                #     print(f"⏭️ File already exists: {save_path}")
                #     download_pdf_links.downloaded_pdfs.add(pdf_url)
                #     continue
                
                try:
                    async with session.get(pdf_url) as resp:
                        if resp.status == 200:
                            # Check file size before downloading
                            content_length = resp.headers.get('content-length')
                            if content_length:
                                file_size_mb = int(content_length) / (1024 * 1024)
                                if file_size_mb > MAX_SIZE_MB:
                                    log_message(f"⚠️ Skipping PDF larger than {MAX_SIZE_MB}MB: {pdf_url} (Size: {file_size_mb:.2f}MB)", "INFO")
                                    continue
                            
                            # Validate content type for PDF files
                            content_type = resp.headers.get('content-type', '').lower()
                            is_pdf_content = (
                                'application/pdf' in content_type or
                                'pdf' in content_type or
                                content_type.startswith('application/octet-stream') or
                                content_type.startswith('binary/')
                            )
                            
                            # Read the content to check for duplicates
                            content = await resp.read()
                            
                            # Check file size after reading content (fallback for servers that don't provide content-length)
                            content_size_mb = len(content) / (1024 * 1024)
                            if content_size_mb > MAX_SIZE_MB:
                                log_message(f"⚠️ Skipping PDF larger than {MAX_SIZE_MB}MB: {pdf_url} (Size: {content_size_mb:.2f}MB)", "INFO")
                                continue
                            
                            # Additional validation: check if content starts with PDF magic bytes
                            if not is_pdf_content and len(content) >= 4:
                                pdf_magic_bytes = b'%PDF'
                                if not content.startswith(pdf_magic_bytes):
                                    log_message(f"⚠️ Skipping non-PDF content from {pdf_url} (Content-Type: {content_type})", "INFO")
                                    continue
                            
                            # Check if content is identical to any existing PDF
                            content_hash = hashlib.md5(content).hexdigest()
                            # if hasattr(download_pdf_links, 'content_hashes') and content_hash in download_pdf_links.content_hashes:
                            #     print(f"⏭️ Skipping duplicate content: {filename}")
                            #     download_pdf_links.downloaded_pdfs.add(pdf_url)
                            #     continue
                            
                            # Store content hash and download the file
                            if not hasattr(download_pdf_links, 'content_hashes'):
                                download_pdf_links.content_hashes = set()
                            download_pdf_links.content_hashes.add(content_hash)
                            
                            # Ensure filename has .pdf extension if it doesn't already
                            if not filename.lower().endswith('.pdf'):
                                filename += '.pdf'
                                save_path = os.path.join(productPath, filename)
                            
                            async with aiofiles.open(save_path, "wb") as f:
                                await f.write(content)
                            
                            # Mark this URL as downloaded with its file path
                            download_pdf_links.downloaded_pdfs[pdf_url] = save_path


                            # AI-powered PDF cleaning with web interface logging
                            log_message(f"🧹 Starting PDF cleaning for: {os.path.basename(save_path)}", "INFO")
                            try:
                                pdf_processing(file_path=save_path, api_key=api_key, log_callback=log_message)
                                log_message(f"✨ PDF cleaning completed: {os.path.basename(save_path)}", "INFO")
                            except Exception as clean_error:
                                log_message(f"⚠️ PDF cleaning failed for {os.path.basename(save_path)}: {str(clean_error)}", "WARNING")
                                log_message(f"📄 Original PDF preserved: {os.path.basename(save_path)}", "INFO")
                        else:
                            log_message(f"❌ Failed to download: {pdf_url} (Status: {resp.status})", "ERROR")
                except Exception as e:
                    log_message(f"❌ Error downloading {pdf_url}: {str(e)}", "ERROR")
    except Exception as e:
        log_message(f"⚠️ Error during PDF processing for {product_url}: {e}", "ERROR")

def detect_pagination_type(url: str) -> str:
    """
    Detect the pagination type from a URL.
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        str: The detected pagination type ("path", "page", "offset", "start", "skip", "limit_offset", "cursor", "after", "before", or "unknown")
    """
    try:
        parsed = urlparse(url)
        path = parsed.path
        query_params = parse_qs(parsed.query)
        
        # Check for path-based pagination
        path_parts = [part for part in path.split('/') if part]
        for i, part in enumerate(path_parts):
            if part.isdigit():
                # Check if this looks like a page number
                if (i > 0 and path_parts[i-1].lower() in ['page', 'p', 'pg', 'products', 'category', 'catalog']) or \
                   (i == len(path_parts) - 1 and len(path_parts) > 1):
                    return "path"
        
        # Check query parameters
        pagination_params = {
            'page': ['page', 'p', 'pg', 'page_num', 'page_number', 'pageNumber'],
            'offset': ['offset'],
            'start': ['start'],
            'skip': ['skip'],
            'limit_offset': ['limit'],
            'cursor': ['cursor'],
            'after': ['after'],
            'before': ['before']
        }
        
        for pagination_type, params in pagination_params.items():
            for param in params:
                if param in query_params:
                    if pagination_type == 'limit_offset' and 'offset' in query_params:
                        return 'limit_offset'
                    return pagination_type
        
        return "unknown"
        
    except Exception as e:
        log_message(f"⚠️ Error detecting pagination type from URL '{url}': {e}", "ERROR")
        return "unknown"


def get_page_number(base_url: str): 
    """
    Enhanced function to extract page number from various URL formats:
    - Path-based: /products/1, /category/page/2
    - Query-based: ?page=1, ?offset=20
    - Hybrid: /products/1?sort=name
    """
    try:
        parsed = urlparse(base_url)
        path = parsed.path
        query_params = parse_qs(parsed.query)
        
        # First, check for path-based pagination (e.g., /products/1, /category/page/2)
        path_parts = [part for part in path.split('/') if part]
        
        # Look for numeric page numbers in the path
        for i, part in enumerate(path_parts):
            if part.isdigit():
                # Check if this looks like a page number (not an ID)
                # Common patterns: /page/1, /1, /p/1, /products/1
                if (i > 0 and path_parts[i-1].lower() in ['page', 'p', 'pg', 'products', 'category', 'catalog']) or \
                   (i == len(path_parts) - 1 and len(path_parts) > 1):
                    log_message(f"📄 Found path-based page number: {part} in URL path", "INFO")
                    return int(part)
        
        # If no path-based pagination found, check query parameters
        pagination_params_to_check = [
            'page', 'p', 'pg', 'page_num', 'page_number', 'pageNumber',
            'offset', 'start', 'skip', 'from',
            'limit', 'size', 'per_page', 'items_per_page',
            'cursor', 'after', 'before', 'next', 'prev',
            'page_id', 'pageid', 'pageno', 'pagenum'
        ]
        
        for param in pagination_params_to_check:
            if param in query_params:
                try:
                    page_num = int(query_params[param][0])
                    log_message(f"📄 Found query-based page number: {param}={page_num}", "INFO")
                    return page_num
                except (ValueError, IndexError):
                    continue
        
        # No pagination found
        log_message(f"⚠️ No pagination detected in URL: {base_url}", "INFO")
        return None
        
    except Exception as e: 
        log_message(f"⚠️ Error extracting page number from URL '{base_url}': {e}", "ERROR")
        return None


def append_page_param(base_url: str, page_number: int, pagination_type: str = "auto") -> str:
    """
    Enhanced pagination parameter handler that supports multiple pagination patterns:
    - Path-based: /products/1 → /products/2
    - Query-based: ?page=1 → ?page=2
    - Hybrid: /products/1?sort=name → /products/2?sort=name
    
    Args:
        base_url (str): The base URL to append pagination to
        page_number (int): The page number to navigate to
        pagination_type (str): Type of pagination to use. Options:
            - "auto": Automatically detect pagination type from URL
            - "path": Path-based pagination (/page/X)
            - "query": Query-based pagination (?page=X)
            - "page": Page-based pagination (?page=X)
            - "offset": Offset-based pagination (?offset=X)
            - "start": Start-based pagination (?start=X)
            - "skip": Skip-based pagination (?skip=X)
            - "limit_offset": Limit-offset pagination (?limit=20&offset=X)
            - "cursor": Cursor-based pagination (?cursor=X)
            - "after": After-based pagination (?after=X)
            - "before": Before-based pagination (?before=X)
    
    Returns:
        str: URL with appropriate pagination parameter
    """
    try: 


        if page_number == None: 
            return base_url
            
        # Get pagination configuration
        pagination_config = DEFAULT_CONFIG.get("pagination_settings", {})
        items_per_page = pagination_config.get("items_per_page", 20)
        
        # Parse the URL
        parsed = urlparse(base_url)
        path = parsed.path
        query_params = parse_qs(parsed.query)
        
        # Detect current pagination type
        current_page = get_page_number(base_url)
        detected_pagination_type = "unknown"
        
        # Check for path-based pagination
        path_parts = [part for part in path.split('/') if part]
        for i, part in enumerate(path_parts):
            if part.isdigit() and current_page:
                # Check if this looks like a page number
                if (i > 0 and path_parts[i-1].lower() in ['page', 'p', 'pg', 'products', 'category', 'catalog']) or \
                   (i == len(path_parts) - 1 and len(path_parts) > 1):
                    detected_pagination_type = "path"
                    break
        
        # If no path-based pagination, check query parameters
        if detected_pagination_type == "unknown":
            pagination_params_to_check = [
                'page', 'p', 'pg', 'page_num', 'page_number', 'pageNumber',
                'offset', 'start', 'skip', 'from',
                'limit', 'size', 'per_page', 'items_per_page',
                'cursor', 'after', 'before', 'next', 'prev',
                'page_id', 'pageid', 'pageno', 'pagenum'
            ]
            
            for param in pagination_params_to_check:
                if param in query_params:
                    detected_pagination_type = param
                    break
        
        # Use detected type if auto is specified
        if pagination_type == "auto":
            pagination_type = detected_pagination_type
            # If still unknown, use fallback
            if pagination_type == "unknown":
                pagination_type = pagination_config.get("fallback_type", "page")
        
        # Handle path-based pagination
        if pagination_type == "path" or (detected_pagination_type == "path" and pagination_type == "auto"):
            # Find the page number in the path and replace it
            new_path_parts = []
            page_replaced = False
            
            for i, part in enumerate(path_parts):
                if part.isdigit() and not page_replaced:
                    # Check if this looks like a page number
                    if (i > 0 and path_parts[i-1].lower() in ['page', 'p', 'pg', 'products', 'category', 'catalog']) or \
                       (i == len(path_parts) - 1 and len(path_parts) > 1):
                        new_path_parts.append(str(page_number))
                        page_replaced = True
                    else:
                        new_path_parts.append(part)
                else:
                    new_path_parts.append(part)
            
            # If no page number found in path, append it
            if not page_replaced:
                new_path_parts.append(str(page_number))
            
            new_path = '/' + '/'.join(new_path_parts)
            new_parsed = parsed._replace(path=new_path)
            
            log_message(f"🔄 Path-based pagination: {path} → {new_path} (page {page_number})", "INFO")
            return urlunparse(new_parsed)
        
        # Handle query-based pagination
        else:
            # Remove any existing pagination parameters
            pagination_params_to_remove = [
                'page', 'p', 'pg', 'page_num', 'page_number', 'pageNumber',
                'offset', 'start', 'skip', 'from',
                'limit', 'size', 'per_page', 'items_per_page',
                'cursor', 'after', 'before', 'next', 'prev',
                'page_id', 'pageid', 'pageno', 'pagenum'
            ]
            
            # Find and remove existing pagination parameter
            existing_pagination_param = None
            for param in pagination_params_to_remove:
                if param in query_params:
                    existing_pagination_param = param
                    query_params.pop(param, None)
                    break
            
            # Calculate pagination values based on type
            if pagination_type in ["page", "p", "pg", "page_num", "page_number", "pageNumber"]:
                query_params['page'] = [str(page_number)]
            elif pagination_type == "offset":
                # Calculate offset based on page number
                offset_value = (page_number - 1) * items_per_page
                query_params['offset'] = [str(offset_value)]
            elif pagination_type == "start":
                # Calculate start based on page number
                start_value = (page_number - 1) * items_per_page
                query_params['start'] = [str(start_value)]
            elif pagination_type == "skip":
                # Calculate skip based on page number
                skip_value = (page_number - 1) * items_per_page
                query_params['skip'] = [str(skip_value)]
            elif pagination_type == "limit_offset":
                query_params['limit'] = [str(items_per_page)]
                # Calculate offset based on page number
                offset_value = (page_number - 1) * items_per_page
                query_params['offset'] = [str(offset_value)]
            elif pagination_type == "cursor":
                # For cursor-based, we'll use a simple numeric cursor
                # In real scenarios, you might need to get the actual cursor from previous page
                query_params['cursor'] = [str(page_number * items_per_page)]
            elif pagination_type == "after":
                query_params['after'] = [str(page_number * items_per_page)]
            elif pagination_type == "before":
                query_params['before'] = [str(page_number * items_per_page)]
            else:
                # Default to page-based pagination
                query_params['page'] = [str(page_number)]
            
            # Reconstruct the URL
            new_query = urlencode(query_params, doseq=True)
            new_parsed = parsed._replace(query=new_query)
            
            log_message(f"🔄 Query-based pagination: {existing_pagination_param or 'page'}={page_number}", "INFO")
            return urlunparse(new_parsed)
            
    except Exception as e:
        log_message(f"⚠️ Error during pagination parameter handling: {e}", "ERROR")
        return base_url


def get_browser_config() -> BrowserConfig:
    """
    Returns the browser configuration for the crawler.

    Returns:
        BrowserConfig: The configuration settings for the browser.
    """

    ua = UserAgent()
    user_agent = ua.random  # Generate a random user agent string
    # https://docs.crawl4ai.com/core/browser-crawler-config/
    return BrowserConfig(
        browser_type="chromium",  # Type of browser to simulate
        headless=True,  # Whether to run in headless mode (no GUI)
        verbose=True,  # Enable verbose logging
        user_agent = user_agent,  # Custom headers to include
        extra_args=[
            "--no-sandbox",
            "--disable-dev_shm-usage",
            "--disable-gpu",
            "--disable-web-security",
            "--disable-features=VizDisplayCompositor",
            "--max_old_space_size=4096",
        ]
    )


def get_regex_strategy() -> RegexExtractionStrategy:
    """
    Returns the configuration for the regex extraction strategy.

    Returns:
        RegexExtractionStrategy: The settings for how to extract data using regex.
    """
    # https://docs.crawl4ai.com/api/strategies/#regexextractionstrategy
    return RegexExtractionStrategy(
        #regex=r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',  # Regex pattern to match links
        pattern= RegexExtractionStrategy.Url,
    )



def get_llm_strategy(api_key: str = None, model: str = "groq/llama-3.1-8b-instant") -> LLMExtractionStrategy:
    """
    Returns the configuration for the language model extraction strategy.
    
    Args:
        api_key (str): The API key for the LLM provider. If None, will try to get from environment.
        model (str): The LLM model to use. Defaults to "groq/llama-3.1-8b-instant".
    
    Returns:
        LLMExtractionStrategy: The settings for how to extract data using LLM.
    """
    # Use provided API key or fall back to environment variable
    if api_key is None:
        api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        raise ValueError("API key is required. Please provide an API key or set GROQ_API_KEY environment variable.")
    
    # https://docs.crawl4ai.com/api/strategies/#llmextractionstrategy
    return LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider = model, # LLM model to use
            api_token= api_key,  # API token for the LLM provider    
        ),
        schema=Venue.model_json_schema(),  # JSON schema of the data model
        extraction_type="schema",  # Type of extraction to perform
        instruction=(
            "You are given HTML content that has already been filtered to include only the main product listing elements from an industrial or e-commerce website. This content was selected using a CSS selector, so it should primarily contain product cards, tiles, or grid items.\n"
            "\n"
            "Your task is to extract all valid product entries from this filtered HTML. For each product, extract the following fields:\n"
            "- productName: The complete product name or title, exactly as displayed on the website, if it is showed more than one time, do not repeat product names more than once\n"
            "- productLink: The full, absolute URL to the product detail page, taken from the href attribute of an anchor tag within the product element.\n"
            "\n"
            "Extraction Rules:\n"
            "- Only include entries that have both a productName and a productLink.\n"
            "- Use the exact product name as shown in the HTML—do not paraphrase or guess.\n"
            "- For productLink, always use the value from the href attribute and convert relative URLs to absolute URLs using the website domain.\n"
            "- Do not include categories, collections, or non-product items.\n"
            "- Do not guess or invent missing data; only extract what is present in the HTML.\n"
            "- Output a list of dictionaries, each matching the required schema.\n"
            "- If the url is incomplete, complete the url with the domain"
            
            "Context:\n"
            "The HTML you receive is already focused on product elements, so you do not need to search the entire page—just extract structured product data from the provided content.\n"
            "Output a list of dictionaries, each matching the required schema.\n"

        ),
        input_format="markdown",  # Format of the input content
        verbose=False,  # Enable verbose logging
    )

def get_pdf_llm_strategy(api_key: str = None, model: str = "groq/llama-3.1-8b-instant") -> LLMExtractionStrategy:
    """
    Returns the configuration for the PDF extraction strategy using LLM.
    
    Args:
        api_key (str): The API key for the LLM provider. If None, will try to get from environment.
        model (str): The LLM model to use. Defaults to "groq/llama-3.1-8b-instant".
    
    Returns:
        LLMExtractionStrategy: The settings for how to extract PDF links using LLM.
    """
    # Use provided API key or fall back to environment variable
    if api_key is None:
        api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        raise ValueError("API key is required. Please provide an API key or set GROQ_API_KEY environment variable.")
    
    # https://docs.crawl4ai.com/api/strategies/#llmextractionstrategy
    return LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider = model, # LLM model to use
            api_token= api_key,  # API token for the LLM provider    
        ),
        schema=PDF.model_json_schema(),
        extraction_type="schema",  # Type of extraction to perform
        instruction=(
               "You are a technical document extraction specialist. Your task is to analyze HTML content and extract PDF download links for technical documents.\n\n"
    
                "OBJECTIVE:\n"
                "Extract downloadable English PDF documents that are:\n"
                "• Only English documents if the document is not in English, Download in the given language\n"
                "• if the same document is available in multiple languages, download only the English version\n"
                "• Data Sheets (product specifications, technical data sheets) \n"
                "• Technical Drawings (dimensional drawings, CAD drawings, schematics)\n"
                "• User Manuals (user manuals, guides)\n"
                "• Product Catalogs (product catalogs, brochures)\n\n"
                
                
                
                "REQUIRED OUTPUT FIELDS:\n"
                "For each valid document, provide:\n"
                "• url: Complete download URL (convert relative URLs to absolute using the domain)\n"
                "• text: Make sure the pdf text is not empty and That is suitable for a path in the file system\n"
                "• type: Must be one of: \"Data Sheet\", \"Technical Drawing\", \"Catalog\", or \"User Manual\"\n"
                "• language: Document language code (\"EN\", \"DE\", \"TR\", etc.) or \"Unknown\"\n"
                "• priority: \"High\" for Data Sheet/Technical Drawing/User Manual, \"Medium\" for Catalog\n\n"
                
                "EXTRACTION GUIDELINES:\n"
                "✓ Look for <a> tags with href attributes pointing to downloadable documents\n"
                "✓ Check for keywords: datasheet, specifications, drawings, catalog, brochure, manuals\n"
                "✓ Extract exact text content - do not modify or paraphrase\n"
                "✓ Convert relative URLs to absolute format\n"
                "✓ Remove duplicates - same URL should appear only once\n"
                "✓ Focus only on the four specified document types\n\n"
                
                "WHAT TO IGNORE:\n"
                "✗ Certificates, certifications, compliance documents\n"
                "✗ Software downloads, apps, tools\n"
                "✗ Marketing materials, press releases\n"
                "✗ Any non-PDF content\n\n"
                
                "OUTPUT FORMAT:\n"
                "Return a JSON array of objects matching the schema. If no valid documents are found, return an empty array [].\n\n"
    
        ),
        input_format="markdown",  # Format of the input content
        verbose=False,  # Enable verbose logging
    )


async def fetch_and_process_page(
    crawler: AsyncWebCrawler,
    css_selector: str,
    page_number: int,
    url: str,
    llm_strategy: LLMExtractionStrategy,
    session_id: str,
    required_keys: List[str],
    seen_names: Set[str],
) -> Tuple[List[dict], bool]:
    """
    Fetches and processes a single page of venue data.

    Args:
        crawler (AsyncWebCrawler): The web crawler instance.
        page_number (int): The page number to fetch.
        base_url (str): The base URL of the website.
        llm_strategy (LLMExtractionStrategy): The LLM extraction strategy.
        session_id (str): The session identifier.
        required_keys (List[str]): List of required keys in the venue data.
        seen_names (Set[str]): Set of venue names that have already been seen.

    Returns:
        Tuple[List[dict], bool]:
            - List[dict]: A list of processed products from the page.
            - bool: A flag indicating if the "No Results Found" message was encountered.
    """

    # Debugging: Print the URL being fetched
    log_message(f"🔄 Crawling page {page_number} from URL: {url}", "INFO")    
    # Fetch page content with the extraction strategy
    result = await crawler.arun(
        url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,  # Do not use cached data
            extraction_strategy=llm_strategy,  # Strategy for data extraction
            target_elements = css_selector,  # Target specific content on the page
            session_id=session_id,  # Unique session ID for the crawl
            scan_full_page=True,
            remove_overlay_elements=True,
            verbose=True,  # Enable verbose logging

        ),
    )

    if not (result.success and result.extracted_content):
        log_message(f"Error fetching page {page_number}: {result.error_message}", "INFO")
        return [], False

    # Parse extracted content
    extracted_data = json.loads(result.extracted_content)
    log_message(f"Type of extracted data: {type(extracted_data)}", "INFO")
    if not extracted_data:
        log_message(f"No products found on page {page_number}.", "INFO")
        return [], False

    
    # Process product
    complete_venues = []
    for venue in extracted_data:
        # Debugging: Print each venue to understand its structure
        log_message(f"{venue}", "PROCESSING")

        # Ignore the 'error' key if it's False
        if venue.get("error") is False:
            venue.pop("error", None)  # Remove the 'error' key if it's False

        if not is_complete_venue(venue, required_keys):
            continue  # Skip incomplete venues

        if is_duplicate_venue(venue["productName"], seen_names):
            log_message(f"Duplicate venue '{venue['productName']}' found. Skipping.", "INFO")
            continue  # Skip duplicate venues

        # Add venue to the list
        seen_names.add(venue["productName"])

        if "productLink" in venue:
            venue["productLink"] = venue["productLink"].replace("/en/en/", "/en/")

            complete_venues.append(venue)

    if not complete_venues:
        log_message(f"No complete venues found on page {page_number}.", "INFO")
        return [], False

    log_message(f"Extracted {len(complete_venues)} venues from page {page_number}.", "INFO")

    return complete_venues, False  # Continue crawling

async def fetch_and_process_page_with_js(
    crawler: AsyncWebCrawler,
    page_url: str,
    llm_strategy: LLMExtractionStrategy,
    button_selector: str,
    elements: list,
    required_keys: list,
    seen_names: set,
) -> Tuple[list, bool]:
    """
    JS-based extraction for sites with dynamic pagination. Extracts product rows using JS, then applies LLM extraction per page.
    Returns (venues, no_results) just like fetch_and_process_page.
    """


    js_commands = f"""
        console.log('[JS] Starting data extraction...');
        let allRowsData = [];
        const rowSelectors = '{", ".join(elements)}';
        const buttonSelector = '{button_selector}';
        const maxPages = 500;
        let currentPage = 1;


        // Helper function to extract rows
        function extractRows() {{
            const pageData = [];
            const rows = document.querySelectorAll(rowSelectors);
            rows.forEach(row => {{
                pageData.push({{
                    html: row.outerHTML
                }});
            }});
            return pageData;
        }}

        // Always extract first page
        allRowsData.push({{
            page: currentPage,
            data: extractRows()
        }});
        console.log(`[JS] Extracted initial page with ${{allRowsData[0].data.length}} rows`);

        await new Promise(r => setTimeout(r, 3000));
        // Only attempt pagination if valid button selector exists
        if (buttonSelector && buttonSelector.trim() !== '') {{
            console.log('[JS] Pagination detected. Starting automatic pagination...');
            let nextButton = document.querySelector(buttonSelector);
            
            let lastPageData = allRowsData[0].data;
            while (currentPage < maxPages && nextButton && nextButton.offsetParent !== null && !nextButton.disabled) {{
                // Click to load next page
                nextButton.click();
                console.log(`[JS] Clicked page ${{currentPage}}`);
                currentPage++;
                
                // Wait for new content to load
                await new Promise(r => setTimeout(r, 9000));
                
                // Extract new page data
                const newPageData = extractRows();

                if(newPageData.length !== lastPageData.length){{
                    console.log('[JS] No new data found. Stopping pagination.');
                    break;
                }}

                allRowsData.push({{
                    page: currentPage,
                    data: newPageData
                }});
                console.log(`[JS] Extracted page ${{currentPage}} with ${{newPageData.length}} rows`);
                
                // Update button reference after DOM changes
                nextButton = document.querySelector(buttonSelector);

                // Update lastPageData
                lastPageData = newPageData;
            }}
            console.log('[JS] Pagination complete');
        }} else {{
            console.log('[JS] No pagination button selector provided. Returning single page data.');
        }}

        try{{
            return allRowsData;
        }}
        catch(error){{
            console.log('[JS] Error: ', error);
            return error.message;
        }}
    """

    complete_venues = []
    try:
        results = await crawler.arun(
            url=page_url,
            config=CrawlerRunConfig(
                extraction_strategy=None,
                cache_mode=CacheMode.BYPASS,
                target_elements=elements,
                scan_full_page=True,
                remove_overlay_elements=True,
                session_id="js_extraction_session",
                js_code=js_commands,
                delay_before_return_html=3.0,
            )
        )

        js_extracted_content = None
        if hasattr(results, 'js_execution_result') and results.js_execution_result:
            # Try to get the results from the JS execution
            if isinstance(results.js_execution_result, dict) and 'results' in results.js_execution_result:
                js_extracted_content = results.js_execution_result['results'][0]
            else:
                js_extracted_content = results.js_execution_result

        if not js_extracted_content:
            log_message("No content extracted via JS", "INFO")
            return [], True
        for items in js_extracted_content:
            log_message(f"processing page: {items['page']}, data length: {len(items['data'])}")
            products_string = "".join([item['html'] for item in items['data']])
            session_id = f"js_extraction_session_{items['page']}"
            raw_html_url = f"raw:\n{page_url}<div>\n{products_string}\n</div>"
            result = await crawler.arun(
                url=raw_html_url,
                config=CrawlerRunConfig(
                    extraction_strategy=llm_strategy,
                    session_id=session_id
                )
            )

            if not result.extracted_content:
                log_message(f"\tNo content extracted for page {items['page']}", "INFO")
                continue
            extracted_content = json.loads(result.extracted_content)
            log_message(f"Extracted {len(extracted_content)} products from page {items['page']}", "INFO")
            new_products = 0
            for product in extracted_content:
                log_message(f"\tprocessing product: {product}", "INFO")
                if product.get("error") is False:
                    product.pop("error", None)
                if not is_complete_venue(product, required_keys):
                    continue
                if is_duplicate_venue(product["productName"], seen_names):
                    log_message(f"\tDuplicate: {product['productName']}", "INFO")
                    continue
                if "productLink" in product:
                    product["productLink"] = product["productLink"].replace("/en/en/", "/en/")
                seen_names.add(product["productName"])
                complete_venues.append(product)
                new_products += 1
            log_message(f"\tAdded {new_products} unique products", "INFO")
        if not complete_venues:
            log_message("\tNo product to scrape", "INFO")
            return [], True
        return complete_venues, False
    except Exception as e:
        log_message(f"Error during JS-based crawling: {str(e)}", "INFO")
        traceback.print_exc()
        return [], True
