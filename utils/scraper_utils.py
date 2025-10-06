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

def generate_product_name_js_commands(primary_selector: str) -> str:
    """
    Generate enhanced JavaScript commands for product name extraction.
    
    Args:
        primary_selector (str): The primary CSS selector from CSV configuration
        
    Returns:
        str: JavaScript code for product name extraction
    """
    return f"""
        console.log('[JS] Starting enhanced product name extraction...');
        await new Promise(r => setTimeout(r, 3000));

        // Primary selector from CSV configuration
        var primarySelector = '{primary_selector}';
        
        // Fallback selectors for common product name patterns
        var fallbackSelectors = [
            'h1',
            '.product-title',
            '.product-name',
            '[data-product-name]',
            '.title',
            'h2',
            '.product-header h1',
            '.product-info h1',
            '.product-details h1',
            '[data-name]',
            '.product-name h1',
            '.product-title h1',
            '.main-title',
            '.page-title',
            'h1.product-title',
            'h1.product-name'
        ];
        
        // Combine primary and fallback selectors
        var allSelectors = [primarySelector, ...fallbackSelectors];
        
        var productName = "";
        var usedSelector = "";
        var extractionAttempts = [];
        
        // Try each selector until we find a valid product name
        for (var i = 0; i < allSelectors.length; i++) {{
            try {{
                var selector = allSelectors[i];
                console.log('[JS] Trying selector:', selector);
                
                var element = document.querySelector(selector);
                if (element) {{
                    var text = element.innerText || element.textContent || "";
                    text = text.trim();
                    
                    // Log extraction attempt
                    extractionAttempts.push({{selector: selector, found: true, text: text}});
                    
                    // Validate the extracted text
                    if (text && text.length > 2 && text.length < 200) {{
                        // Clean the text
                        productName = text
                        if (productName.length > 2) {{
                            usedSelector = selector;
                            console.log('[JS] Successfully extracted product name:', productName);
                            console.log('[JS] Used selector:', selector);
                            break;
                        }}
                    }}
                }} else {{
                    extractionAttempts.push({{selector: selector, found: false, text: ""}});
                }}
            }} catch (error) {{
                console.log('[JS] Error with selector', allSelectors[i], ':', error.message);
                extractionAttempts.push({{selector: allSelectors[i], found: false, error: error.message}});
                continue;
            }}
        }}
        
        // Final validation and fallback
        if (!productName || productName.length < 2) {{
            console.log('[JS] No valid product name found, using fallback');
            console.log('[JS] Extraction attempts:', JSON.stringify(extractionAttempts, null, 2));
            productName = "Unnamed Product";
            usedSelector = "fallback";
        }}
        
        console.log('[JS] Final product name:', productName);
        console.log('[JS] Selector used:', usedSelector);
        console.log('[JS] Total attempts made:', extractionAttempts.length);
        
        return productName;
        """

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
    sanitized = product_name.replace('\\', '/').replace('/', '_') # (if you are working with windows uncomment this)
    
    # Replace other invalid characters
    invalid_chars = r'[<>:?*]'
    sanitized = re.sub(invalid_chars, '_', sanitized)
    invalid_chars = r'["]'
    sanitized = re.sub(invalid_chars, '', sanitized)
    
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


def infer_extension_from_url_and_headers(url: str, headers: dict = None) -> str:
    """
    Infer file extension from URL and HTTP headers.
    
    Args:
        url (str): The file URL
        headers (dict): Optional HTTP response headers
        
    Returns:
        str: File extension without dot (e.g., 'pdf', 'zip', 'step') or empty string if unknown
    """
    # Content-type to extension mapping
    content_type_map = {
        'application/pdf': 'pdf',
        'application/zip': 'zip',
        'application/x-zip-compressed': 'zip',
        'application/x-step': 'step',
        'application/step': 'step',
        'model/step': 'step',
        'application/iges': 'iges',
        'model/iges': 'iges',
        'application/acad': 'dwg',
        'application/x-acad': 'dwg',
        'application/x-autocad': 'dwg',
        'application/dwg': 'dwg',
        'application/dxf': 'dxf',
        'application/x-dxf': 'dxf',
        'model/stl': 'stl',
        'application/sla': 'stl',
        'application/x-navistyle': 'stl',
    }
    
    # First, try to get extension from headers
    if headers:
        content_type = headers.get('content-type', '').lower().split(';')[0].strip()
        if content_type in content_type_map:
            return content_type_map[content_type]
    
    # Fall back to URL extension
    parsed = urlparse(url)
    path = parsed.path.lower()
    if '.' in path:
        ext = path.rsplit('.', 1)[-1]
        # Validate extension (max 5 chars, alphanumeric)
        if ext and len(ext) <= 5 and ext.isalnum():
            return ext
    
    return ''


def is_pdf_extension(ext: str) -> bool:
    """
    Check if the given extension is a PDF.
    
    Args:
        ext (str): File extension (with or without dot)
        
    Returns:
        bool: True if extension is PDF
    """
    if not ext:
        return False
    ext_clean = ext.lower().lstrip('.')
    return ext_clean == 'pdf'


def generate_generic_filename_from_llm_text(pdf_text: str, pdf_type: str, pdf_language: str, product_name: str, ext: str) -> str:
    """
    Generate an appropriate filename for any file type using LLM extracted text.
    
    Args:
        pdf_text (str): The text extracted by LLM from the file link
        pdf_type (str): The type of file (Data Sheet, CAD, ZIP, etc.)
        pdf_language (str): The language of the file
        product_name (str): The name of the product for fallback naming
        ext (str): File extension (without dot)
        
    Returns:
        str: A sanitized filename with appropriate extension
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
            r'^file\s*',
            r'^cad\s*',
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
        
        # Combine base filename with type info and extension
        if type_suffix:
            filename = f"{base_filename}{type_suffix}.{ext}"
        else:
            filename = f"{base_filename}.{ext}"
    else:
        # Fallback to product name if no text available
        filename = f"{sanitize_folder_name(product_name)}.{ext}"
    
    # Sanitize the filename
    filename = sanitize_filename(filename)
    
    return filename


# https://www.ors.com.tr/en/tek-sirali-sabit-bilyali-rulmanlar
async def download_pdf_links(
        crawler: AsyncWebCrawler, 
        product_url: str, 
        output_folder: str, 
        pdf_llm_strategy: LLMExtractionStrategy,
        pdf_selector: str | list,
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




    # Global dictionary to track downloaded files with their file paths across all products
    if not hasattr(download_pdf_links, 'downloaded_files'):
        download_pdf_links.downloaded_files = {}  # Dict: {url: file_path} for all file types

    try:
        log_message(f"🔍 Starting file extraction for product page", "INFO")
        log_message(f"📍 Using selectors: {pdf_selector}", "INFO")

        log_message(f"Name Selector: {pdf_selector[1]}", "INFO")


        # Enhanced JavaScript product name extraction using helper function
        # The second selector will be used as primary, with comprehensive fallback selectors
        js_commands = generate_product_name_js_commands(pdf_selector[1])

        product_url = f"{product_url}#documents"
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
                # simulate_user=True,
                js_code=js_commands,
            )
        )

        if not pdf_result.success or not pdf_result.extracted_content:
            log_message(f"❌ Failed to extract file content from product page", "ERROR")
            log_message(f"🔗 Product URL: {product_url}", "INFO")
            return

        # Step 2: Parse extracted content
        try:
            extracted_data = json.loads(pdf_result.extracted_content)

            # with open("cleaned_html.html", "w", encoding="utf-8") as f:
            #     f.write(pdf_result.cleaned_html)
            # with open("fit_html.html", "w", encoding="utf-8") as f:
            #     f.write(pdf_result.fit_html)

        except json.JSONDecodeError as e:
            log_message(f"❌ Failed to parse extracted JSON content: {e}", "ERROR")
            return

        if not extracted_data:
            log_message(f"📭 No files found on product page", "INFO")
            return

        log_message(f"🔗 Found {len(extracted_data)} potential file(s) via CSS selector", "INFO")

        # Enhanced product name extraction with better error handling
        try:
            if (pdf_result.js_execution_result and 
                'results' in pdf_result.js_execution_result and 
                len(pdf_result.js_execution_result['results']) > 0):
                
                productName = pdf_result.js_execution_result['results'][0]
                log_message(f"✅ Product Name Extracted: '{productName}'", "INFO")
                
                # Validate extracted product name
                if productName and productName.strip() and productName != "Unnamed Product":
                    derived_product_name = productName.strip()
                    log_message(f"📝 Using extracted product name: '{derived_product_name}'", "INFO")
                else:
                    derived_product_name = "Unnamed Product"
                    log_message(f"⚠️ Invalid product name extracted, using fallback: '{derived_product_name}'", "WARNING")
            else:
                derived_product_name = "Unnamed Product"
                log_message(f"❌ No JavaScript execution result found, using fallback: '{derived_product_name}'", "ERROR")
                
        except (KeyError, IndexError, TypeError) as e:
            derived_product_name = "Unnamed Product"
            log_message(f"❌ Error extracting product name from JavaScript result: {str(e)}", "ERROR")
            log_message(f"📝 Using fallback product name: '{derived_product_name}'", "INFO")

        # Step 3: Split extracted files into PDFs and other file types
        pdf_docs = []
        other_docs = []
        seen_urls_in_page = set()

        for item in extracted_data:
            file_url = item.get('url', '')
            
            if not file_url:
                log_message(f"⚠️ Skipping item with missing URL: {item}", "WARNING")
                continue
            
            # Check for duplicates within this page
            if file_url in seen_urls_in_page:
                log_message(f"⏭️ Skipping duplicate URL: {file_url}", "INFO")
                continue
            
            seen_urls_in_page.add(file_url)
            
            # Infer extension from URL (we'll refine with headers during download)
            ext = infer_extension_from_url_and_headers(file_url)
            
            # Classify as PDF or other
            if is_pdf_extension(ext):
                # Only add PDFs if we haven't reached the limit
                if len(pdf_docs) < 3:
                    pdf_docs.append(item)
                    log_message(f"✅ Validated PDF ({len(pdf_docs)}/3): {item.get('type', 'Unknown')} - {item.get('text', 'Unknown')}", "INFO")
                else:
                    log_message(f"📊 Skipped PDF (limit reached): {item.get('type', 'Unknown')}", "INFO")
            else:
                # Add all other file types without limit
                other_docs.append(item)
                file_type = item.get('type', 'Unknown')
                log_message(f"✅ Validated {ext.upper() if ext else 'file'}: {file_type} - {item.get('text', 'Unknown')}", "INFO")

        # Check if any files were found
        total_files = len(pdf_docs) + len(other_docs)
        if total_files == 0:
            log_message(f"📭 No files found on page for product: {derived_product_name}", "INFO")
            log_message(f"🔗 Product URL: {product_url}", "INFO")
            return {"productLink": product_url, "productName": derived_product_name, "category": cat_name, "saved_count": 0, "has_datasheet": False}

        log_message(f"📄 Found {len(pdf_docs)} PDF(s) and {len(other_docs)} other file(s) for product: {derived_product_name}", "INFO")

        # Create the download folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Create category-based folder structure: output/{category}/{product}
        sanitized_cat_name = sanitize_folder_name(cat_name)
        category_path = os.path.join(output_folder, sanitized_cat_name)
        os.makedirs(category_path, exist_ok=True)
        
        productPath = os.path.join(category_path, sanitize_folder_name(derived_product_name))
        if not os.path.exists(productPath):
            os.makedirs(productPath)
            log_message(f"📁 Created folder structure: {sanitized_cat_name}/{sanitize_folder_name(derived_product_name)}", "INFO")


        # Enhanced sorting: Priority first, then by document type preference, then by language
        def sort_key(doc):
            priority_score = {'High': 3, 'Medium': 2, 'Low': 1}.get(doc.get('priority', 'Unknown'), 0)
            
            # Type preference: Data Sheet > Technical Drawing > Manual > CAD > Others
            type_score = 0
            doc_type = doc.get('type', '').lower()
            if 'data sheet' in doc_type or 'datasheet' in doc_type or 'specification' in doc_type:
                type_score = 4
            elif 'drawing' in doc_type or 'dimensional' in doc_type:
                type_score = 3
            elif 'manual' in doc_type or 'guide' in doc_type:
                type_score = 2
            elif 'cad' in doc_type or 'step' in doc_type or 'iges' in doc_type or 'dwg' in doc_type:
                type_score = 1
            
            # Language preference: English > German > Turkish > Others
            language_score = 0
            doc_language = doc.get('language', '').lower()
            if 'english' in doc_language:
                language_score = 4
            elif 'german' in doc_language or 'deutsch' in doc_language:
                language_score = 3
            elif 'turkish' in doc_language or 'türkçe' in doc_language:
                language_score = 2
            else:
                language_score = 1
            
            return (priority_score, type_score, language_score)

        # Sort PDFs by priority
        pdf_docs.sort(key=sort_key, reverse=True)
        
        # Sort other docs by priority (CAD files first)
        other_docs.sort(key=sort_key, reverse=True)
        
        # Combine for download processing
        all_files = pdf_docs + other_docs

        log_message(f"📊 Processing {len(all_files)} files ({len(pdf_docs)} PDFs, {len(other_docs)} others) by priority", "INFO")

        # Log selected document summary
        if all_files:
            log_message("📋 Selected documents:", "INFO")
            for i, doc in enumerate(all_files, 1):
                log_message(f"   {i}. {doc.get('type', 'Unknown')} ({doc.get('language', 'Unknown')}) - Priority: {doc.get('priority', 'Unknown')}", "INFO")

        # Download all files with duplicate checking (SSL verification disabled)
        log_message(f"📥 Starting download of {len(all_files)} file(s)", "INFO")
        saved_files = []
        has_datasheet = False
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            for i, file_info in enumerate(all_files, 1):
                file_url = file_info['url']
                file_text = file_info['text']
                file_type = file_info['type']
                file_language = file_info.get('language', 'Unknown')
                file_priority = file_info.get('priority', 'Unknown')
                priority_emoji = "🔴" if file_priority == 'High' else "🟡" if file_priority == 'Medium' else "🟢"
                
                # Infer extension from URL initially
                file_ext = infer_extension_from_url_and_headers(file_url)
                is_pdf = is_pdf_extension(file_ext)
                
                file_type_label = "PDF" if is_pdf else file_ext.upper() if file_ext else "file"
                log_message(f"{priority_emoji} Downloading {file_type_label} {i}/{len(all_files)}", "INFO")
                
                # Check if this exact file URL has been downloaded before (global tracking)
                if file_url in download_pdf_links.downloaded_files:
                    previous_path = download_pdf_links.downloaded_files[file_url]
                    if os.path.exists(previous_path):
                        try:
                            # Generate filename using generic helper
                            filename = generate_generic_filename_from_llm_text(file_text, file_type, file_language, derived_product_name, file_ext or 'bin')
                            filename = filename.replace('\\', '_').replace('/', '_')
                            save_path = os.path.join(productPath, filename)
                            
                            # Copy the previously processed file to current product folder with new filename
                            shutil.copy2(previous_path, save_path)
                            log_message(f"📋 Copied previously downloaded {file_type_label}: {filename}", "INFO")
                            saved_files.append(save_path)
                            # mark datasheet if type matches
                            if file_type and ('data sheet' in file_type.lower() or 'datasheet' in file_type.lower() or 'specification' in file_type.lower()):
                                has_datasheet = True
                            continue
                        except Exception as copy_error:
                            log_message(f"⚠️ Failed to copy file: {str(copy_error)}", "WARNING")
                            log_message(f"   Will re-download and process: {file_url}", "INFO")
                    else:
                        log_message(f"⚠️ Previously downloaded file not found at: {previous_path}", "WARNING")
                        log_message(f"   Will re-download and process: {file_url}", "INFO")
                
                try:
                    async with session.get(file_url) as resp:
                        if resp.status == 200:
                            # Refine extension from response headers
                            refined_ext = infer_extension_from_url_and_headers(file_url, resp.headers)
                            if refined_ext:
                                file_ext = refined_ext
                                is_pdf = is_pdf_extension(file_ext)
                            
                            # Check file size before downloading
                            content_length = resp.headers.get('content-length')
                            if content_length:
                                file_size_mb = int(content_length) / (1024 * 1024)
                                if file_size_mb > MAX_SIZE_MB:
                                    log_message(f"⚠️ Skipping file larger than {MAX_SIZE_MB}MB: {file_url} (Size: {file_size_mb:.2f}MB)", "INFO")
                                    continue
                            
                            # Read the content
                            content = await resp.read()
                            
                            # Check file size after reading content (fallback for servers that don't provide content-length)
                            content_size_mb = len(content) / (1024 * 1024)
                            if content_size_mb > MAX_SIZE_MB:
                                log_message(f"⚠️ Skipping file larger than {MAX_SIZE_MB}MB: {file_url} (Size: {content_size_mb:.2f}MB)", "INFO")
                                continue
                            
                            # Validate content for PDFs: check if content starts with PDF magic bytes
                            if is_pdf and len(content) >= 4:
                                pdf_magic_bytes = b'%PDF'
                                if not content.startswith(pdf_magic_bytes):
                                    log_message(f"⚠️ Skipping non-PDF content from {file_url}", "INFO")
                                    continue
                            
                            # Check if content is identical to any existing file
                            content_hash = hashlib.md5(content).hexdigest()
                            
                            # Store content hash for deduplication
                            if not hasattr(download_pdf_links, 'content_hashes'):
                                download_pdf_links.content_hashes = set()
                            download_pdf_links.content_hashes.add(content_hash)
                            
                            # Generate filename using generic helper with proper extension
                            filename = generate_generic_filename_from_llm_text(file_text, file_type, file_language, derived_product_name, file_ext or 'bin')
                            filename = filename.replace('\\', '_').replace('/', '_')
                            
                            # Ensure filename has proper extension
                            if not filename.lower().endswith(f'.{file_ext}'):
                                filename = f"{filename.rsplit('.', 1)[0]}.{file_ext}" if '.' in filename else f"{filename}.{file_ext}"
                            
                            save_path = os.path.join(productPath, filename)
                            
                            # Save the file
                            async with aiofiles.open(save_path, "wb") as f:
                                await f.write(content)


                            # Post-download certificate filter (only for PDFs)
                            if is_pdf:
                                lowered = os.path.basename(save_path).lower()
                                cert_terms = [
                                    'certificate','certification','certifications','iso','tse', 'declaration', 
                                    'coc','iec','emc','ped','fda','rohs','iecex','csa','warranty'
                                ]
                                if any(term in lowered for term in cert_terms): 
                                    try: 
                                        os.remove(save_path)
                                        log_message(f"⏭️ Removed certificate PDF: {os.path.basename(save_path)}", "INFO")
                                    except Exception as e:
                                        log_message(f"⚠️ Failed to remove certificate PDF: {os.path.basename(save_path)}: {str(e)}", "WARNING")
                                    continue
                            
                            # Mark this URL as downloaded with its file path
                            download_pdf_links.downloaded_files[file_url] = save_path
                            saved_files.append(save_path)
                            
                            # Mark datasheet if type matches
                            if file_type and ('data sheet' in file_type.lower() or 'datasheet' in file_type.lower() or 'specification' in file_type.lower()):
                                has_datasheet = True

                            # AI-powered PDF cleaning (only for PDFs)
                            if is_pdf:
                                log_message(f"🧹 Starting PDF cleaning for: {os.path.basename(save_path)}", "INFO")
                                try:
                                    pdf_processing(file_path=save_path, api_key=api_key, log_callback=log_message)
                                    log_message(f"✨ PDF cleaning completed: {os.path.basename(save_path)}", "INFO")
                                except Exception as clean_error:
                                    log_message(f"⚠️ PDF cleaning failed for {os.path.basename(save_path)}: {str(clean_error)}", "WARNING")
                                    log_message(f"📄 Original PDF preserved: {os.path.basename(save_path)}", "INFO")
                            else:
                                log_message(f"✅ Saved {file_type_label}: {os.path.basename(save_path)}", "INFO")
                        else:
                            log_message(f"❌ Failed to download: {file_url} (Status: {resp.status})", "ERROR")
                except Exception as e:
                    log_message(f"❌ Error downloading {file_url}: {str(e)}", "ERROR")
    except Exception as e:
        log_message(f"⚠️ Error during file processing for {product_url}: {e}", "ERROR")
        return {"productLink": product_url, "productName": None, "category": cat_name, "saved_count": 0, "has_datasheet": False}
    
    # Return summary
    return {
        "productLink": product_url,
        "productName": derived_product_name,
        "category": cat_name,
        "saved_count": len(saved_files),
        "has_datasheet": has_datasheet,
    }
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
            "You are given filtered HTML for product listings (cards/tiles).\n"
            "Extract only product links.\n"
            "For each product, output an object with: productLink (absolute URL from an <a href>).\n"
            "Convert relative URLs to absolute using the page domain.\n"
            "Return a JSON array matching the schema."
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
            "You are given filtered HTML from a product page, including elements for the product name (via provided selectors)"
            " and anchors for downloadable technical documents and CAD files.\n"
            "Extract technical PDFs, CAD files (STEP, IGES, DWG, DXF, STL), ZIP archives, and other downloadable assets. For each document, output: url, text, type, language, priority, productName.\n"
            "- productName: the exact product title text from the second selector, do not guess or infer the product name.\n"
            "    - productName must be an English product name.\n"
            "- url: absolute link to the file. Convert relative links using the page domain.\n"
            "    - Do not add any escape characters to the url.\n"
            "- text: the link text or button text describing the file.\n"
            "- type: one of Data Sheet, Technical Drawing, Catalog, User Manual, CAD, ZIP, STEP, IGES, DWG, DXF, STL, or Generic.\n"
            "- language: language code like EN/DE/TR or Unknown.\n"
            "- priority: High for Data Sheet/Technical Drawing/User Manual/CAD, Medium for Catalog/ZIP.\n"
            "Ignore certificates/compliance-only links. Return a JSON array matching the schema. Ensure all entries use the same complete productName value."
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


    js_commands = f"""
        console.log('[JS] Starting data extraction...');
        await new Promise(r => setTimeout(r, 3000));
        return 0;
    """
    # Debugging: Print the URL being fetched
    log_message(f"🔄 Crawling page {page_number} from URL: {url}", "INFO")    
    # Fetch page content with the extraction strategy
    result = await crawler.arun(
        url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,  # Do not use cached data
            js_code=js_commands,
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

        # Use productLink for duplicate detection
        if is_duplicate_venue(venue.get("productLink", ""), seen_names):
            log_message(f"Duplicate link '{venue.get('productLink','')}' found. Skipping.", "INFO")
            continue  # Skip duplicate venues

        # Add venue to the list
        if "productLink" in venue:
            link = venue["productLink"].replace("/en/en/", "/en/")
            venue["productLink"] = link
            seen_names.add(link)
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
                const newPageData = extractRows()

                allRowsData.push({{
                    page: currentPage,
                    data: newPageData
                }});
                console.log(`[JS] Extracted page ${{currentPage}} with ${{newPageData.length}} rows`);

                if(newPageData.length !== lastPageData.length){{
                    console.log('[JS] No new data found. Stopping pagination.');
                    break;
                }}
                
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
                if is_duplicate_venue(product.get("productLink",""), seen_names):
                    log_message(f"\tDuplicate link: {product.get('productLink','')}", "INFO")
                    continue
                if "productLink" in product:
                    product["productLink"] = product["productLink"].replace("/en/en/", "/en/")
                    seen_names.add(product["productLink"])
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

