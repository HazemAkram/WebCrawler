"""
DeepSeek AI Web Crawler
Copyright (c) 2026 Ayaz Mensyoğlu

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
import random  # For random delays in cookie extraction
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import partial
import multiprocessing

import aiofiles
import aiohttp
from http.cookies import SimpleCookie

from typing import List, Set, Tuple
from fake_useragent import UserAgent

from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import re

# AI-powered PDF processing is now handled automatically in cleaner.py

def make_config_type_filename(file_type, file_language, ext, folder_path):
    allowed_types = DEFAULT_CONFIG.get("pdf_settings", {}).get("allowed_types", [])
    # Find the closest case-insensitive match, fallback to 'Generic'
    config_type = next((t for t in allowed_types if t.lower() == (file_type or '').lower()), "Generic")
    # Normalize for filename
    type_part = config_type.replace(' ', '_')
    lang_part = file_language.upper() if file_language else "UNKNOWN"
    base_name = f"{type_part} ({lang_part})"
    filename = f"{base_name}.{ext}"
    index = 1
    while os.path.exists(os.path.join(folder_path, filename)):
        index += 1
        filename = f"{base_name}_{index}.{ext}"
    return sanitize_filename(filename)

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

# API endpoint configuration
API_BASE_URL = "https://factory-help.online/prods/json"

async def create_unified_browser_context(domain_url: str):
    """
    Create a unified Playwright browser context for both API and HTML crawling.
    This ensures all requests (API calls, HTML navigation, PDF downloads) use
    the same browser fingerprint, cookies, and headers for maximum anti-bot stealth.
    
    Args:
        domain_url: The full URL of the domain to visit (e.g., 'https://example.com')
        
    Returns:
        Dictionary with:
        - 'browser': Playwright browser instance
        - 'context': Browser context with cookies/session
        - 'page': Initial page (already visited domain)
        - 'headers': Extracted browser headers
        - 'user_agent': Real browser user agent
    """
    def _log(message, level="INFO"):
        if log_callback:
            log_callback(message, level)
        else:
            try:
                print(f"[{level}] {message}")
            except UnicodeEncodeError:
                print(f"[{level}] {message.encode('ascii', 'replace').decode('ascii')}")
    
    try:
        from playwright.async_api import async_playwright
        
        _log(f"🌐 Creating unified browser context for: {domain_url}", "INFO")
        
        playwright = await async_playwright().start()
        
        # Launch browser with anti-detection config
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-features=VizDisplayCompositor",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-site-isolation-trials",
                "--allow-insecure-localhost",
                "--window-size=1920,1080",

            ]
        )
        
        # Create context with realistic browser settings
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
            timezone_id='America/New_York',
        )
        
        page = await context.new_page()

        domain_loaded = False
        try:
            # Visit the domain (best-effort). Some domains (e.g. ABB) can intermittently
            # throw net::ERR_HTTP2_PROTOCOL_ERROR in Chromium; we still want to keep
            # the context alive so API calls via context.request can proceed.
            await page.goto(domain_url, timeout=45000, wait_until="domcontentloaded")
            domain_loaded = True
            _log(f"✅ Successfully loaded page: {domain_url}", "INFO")

            # Additional delay to ensure all cookies are set (only when navigation succeeds)
            delay = random.uniform(10, 15)
            _log(f"⏳ Waiting {delay:.1f} seconds for cookies to be set...", "INFO")
            await asyncio.sleep(delay)
        except Exception as page_error:
            _log(f"⚠️ Error loading page {domain_url}: {str(page_error)}", "WARNING")
            _log("➡️ Proceeding without homepage navigation; API requests may still work but cookies may be empty.", "WARNING")

        # Extract cookies from the browser context (may be empty if navigation failed)
        cookies = await context.cookies()
        cookies_dict = {cookie['name']: cookie['value'] for cookie in cookies}

        # Get the user agent from the page; if page is still about:blank, fallback to configured UA
        try:
            user_agent = await page.evaluate("navigator.userAgent")
        except Exception:
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"

        # Prepare canonical headers that mimic a real browser
        # These will be used for ALL requests (API, HTML, PDF downloads)
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': domain_url,
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-User': '?1',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
        }

        _log(f"🍪 Extracted {len(cookies_dict)} cookies from unified browser session", "INFO")
        _log(f"🔧 Prepared canonical headers for all requests", "INFO")

        return {
            'playwright': playwright,
            'browser': browser,
            'context': context,
            'page': page,
            'headers': headers,
            'user_agent': user_agent,
            'cookies': cookies_dict,
            'domain_loaded': domain_loaded,
        }
            
    except ImportError:
        _log("⚠️ Playwright not available", "WARNING")
        return None
    except Exception as e:
        _log(f"❌ Unexpected error creating browser context: {str(e)}", "ERROR")
        return None

async def close_unified_browser_context(browser_ctx: dict):
    """Close the unified browser context and cleanup resources."""
    if browser_ctx:
        try:
            if 'page' in browser_ctx and browser_ctx['page']:
                await browser_ctx['page'].close()
            if 'context' in browser_ctx and browser_ctx['context']:
                await browser_ctx['context'].close()
            if 'browser' in browser_ctx and browser_ctx['browser']:
                await browser_ctx['browser'].close()
            if 'playwright' in browser_ctx and browser_ctx['playwright']:
                await browser_ctx['playwright'].stop()
        except Exception as e:
            if log_callback:
                log_callback(f"⚠️ Error closing browser context: {str(e)}", "WARNING")

async def get_browser_cookies_for_domain(domain_url: str) -> dict:
    """
    Use Playwright to visit a domain and extract cookies to bypass bot detection.
    
    Args:
        domain_url: The full URL of the domain to visit (e.g., 'https://example.com')
        
    Returns:
        Dictionary with 'cookies' (dict) and 'headers' (dict) for use in requests
    """
    def _log(message, level="INFO"):
        if log_callback:
            log_callback(message, level)
        else:
            try:
                print(f"[{level}] {message}")
            except UnicodeEncodeError:
                # Fallback for Windows console encoding issues
                print(f"[{level}] {message.encode('ascii', 'replace').decode('ascii')}")
    
    try:
        from playwright.async_api import async_playwright
        
        _log(f"🌐 Opening browser to extract cookies from: {domain_url}", "INFO")
        
        async with async_playwright() as p:
            # Launch browser with similar config to get_browser_config
            browser = await p.chromium.launch(
                headless=True,
                args=[
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "--no-sandbox",
                "--disable-dev_shm-usage",
                "--disable-gpu",
                "--disable-features=VizDisplayCompositor",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-site-isolation-trials",
                "--allow-insecure-localhost",
                "--window-size=1920,1080",
                ]
            )
            
            # Create context with realistic browser settings
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
                viewport={'width': 1080, 'height': 720},
                locale='en-US',
                timezone_id='America/New_York',
            )
            
            page = await context.new_page()
            
            try:
                # Visit the domain and wait for it to fully load
                await page.goto(domain_url, wait_until='networkidle', timeout=30000)
                _log(f"✅ Successfully loaded page: {domain_url}", "INFO")
                
                # Additional delay to ensure all cookies are set (especially dynamic ones)
                # Some sites set cookies via JavaScript after page load
                delay = random.uniform(10, 15)  # Random delay between 3-5 seconds
                _log(f"⏳ Waiting {delay:.1f} seconds for cookies to be set...", "INFO")
                await asyncio.sleep(delay)
                
                # Extract cookies from the browser context
                cookies = await context.cookies()
                
                # Format cookies for aiohttp (dict format)
                cookies_dict = {cookie['name']: cookie['value'] for cookie in cookies}
                
                # Get the user agent from the page
                user_agent = await page.evaluate("navigator.userAgent")
                
                # Prepare headers that mimic a real browser
                headers = {
                    'User-Agent': user_agent,
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Referer': domain_url,
                    'Origin': domain_url,
                    'Connection': 'keep-alive',
                    'Sec-Fetch-Dest': 'empty',
                    'Sec-Fetch-Mode': 'cors',
                    'Sec-Fetch-Site': 'same-origin',
                }
                
                _log(f"🍪 Extracted {len(cookies_dict)} cookies from browser session", "INFO")
                
                await browser.close()
                
                return {
                    'cookies': cookies_dict,
                    'headers': headers
                }
                
            except Exception as page_error:
                _log(f"⚠️ Error loading page {domain_url}: {str(page_error)}", "WARNING")
                await browser.close()
                # Return empty cookies/headers on error
                return {
                    'cookies': {},
                    'headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
                        'Accept': 'application/json, text/javascript, */*; q=0.01',
                    }
                }
                
    except ImportError:
        _log("⚠️ Playwright not available, proceeding without cookie extraction", "WARNING")
        return {
            'cookies': {},
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
            }
        }
    except Exception as e:
        _log(f"❌ Unexpected error in cookie extraction: {str(e)}", "ERROR")
        return {
            'cookies': {},
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
            }
        }

async def fetch_products_from_api_via_browser(
    domain_name: str, 
    page_number: int,
    browser_context: dict,
    start_from_product_index: int = 0,
) -> tuple[List[dict], bool]:
    """
    Fetch products for a given domain and page from the API using Playwright browser context.
    This ensures the API request uses the same browser fingerprint as HTML crawling.
    
    Args:
        domain_name: The domain name to fetch products for
        page_number: The page number to fetch
        browser_context: Unified browser context from create_unified_browser_context()
        start_from_product_index: Index to start from (0-based). Products before this index will be skipped.
                                   Useful for resuming failed crawls or processing specific product ranges.
        
    Returns:
        List of product dicts with 'productName' and 'productLink' keys (starting from specified index)
    """
    def _log(message, level="INFO"):
        if log_callback:
            log_callback(message, level)
        else:
            print(f"[{level}] {message}")
    
    products = []
    
    try:
        context = browser_context['context']
        headers = browser_context['headers']
        is_empty = True
        
        # Construct API URL
        url = f"{API_BASE_URL}?domain={domain_name}&page={page_number}"
        _log(f"📡 Fetching products from API via browser: {url}", "INFO")
        
        # Use Playwright's context.request to make the API call
        # This ensures the request uses the same browser session/cookies/headers
        api_request = context.request
        
        try:
            response = await api_request.get(
                url,
                headers={
                    **headers,
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                },
                timeout=30000
            )
            
            if response.status != 200:
                _log(f"⚠️ API returned status {response.status} for {domain_name} page {page_number}", "WARNING")
                return [], True
            
            data = await response.json()
            
            # if data:
            #     is_empty = False
            #     print("data is not empty")
            # else:
            #     print("data is empty")


            # Check if response is empty or not a list
            if not data or not isinstance(data, list):
                _log(f"✅ API pagination complete for {domain_name} at page {page_number} (empty response)", "INFO")
                return [], True
            else: 
                is_empty = False
            
            # Extract products from response
            for idx, item in enumerate(data):
                name = item.get("name", "").strip()
                referenceNumber = item.get("id_product","")
                source_url = item.get("source_url", "").strip()
                with_pdf = item.get("with_pdf", '0')
                
                if with_pdf == '0': 
                    if name and source_url:
                        products.append({
                            "productName": name,
                            "productLink": source_url,
                            "referenceNumber": referenceNumber,
                        })
                    else:
                        _log(f"⚠️ Skipping malformed product entry: name='{name}', url='{source_url}'", "WARNING")
                else:
                    _log(f"⚠️ Skipping product entry with PDF: name='{name}'.", "WARNING")
            # Apply start_from_product_index filter
            total_fetched = len(products)
            if start_from_product_index > 0:
                if start_from_product_index >= total_fetched:
                    _log(f"⚠️ start_from_product_index ({start_from_product_index}) >= total products ({total_fetched}), returning empty list", "WARNING")
                    return [], True
                
                skipped_count = start_from_product_index
                products = products[start_from_product_index:]
                _log(f"📦 Fetched {total_fetched} products from page {page_number}, skipped first {skipped_count}, returning {len(products)} products", "INFO")
            else:
                _log(f"📦 Fetched {len(products)} products from page {page_number}", "INFO")
            
        except Exception as e:
            _log(f"❌ Error fetching page {page_number} for {domain_name}: {str(e)}", "ERROR")
            return [], True
    
    except Exception as e:
        _log(f"❌ Unexpected error in API fetch: {str(e)}", "ERROR")
        return [], True
    
    return products, is_empty

async def fetch_products_from_api(domain_name: str, session: aiohttp.ClientSession = None) -> List[dict]:
    """
    Fetch all products for a given domain from the API using pagination.
    
    Args:
        domain_name: The domain name to fetch products for
        session: Optional aiohttp session (will create one if not provided)
        
    Returns:
        List of product dicts with 'productName' and 'productLink' keys
    """
    def _log(message, level="INFO"):
        if log_callback:
            log_callback(message, level)
        else:
            print(f"[{level}] {message}")
    
    products = []
    page = 1
    close_session = False
    
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True
    
    try:
        while True:
            url = f"{API_BASE_URL}?domain={domain_name}&page={page}"
            _log(f"📡 Fetching products from API: {url}", "INFO")
            
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        _log(f"⚠️ API returned status {response.status} for {domain_name} page {page}", "WARNING")
                        break
                    
                    data = await response.json()
                    
                    # Check if response is empty or not a list
                    if not data or not isinstance(data, list):
                        _log(f"✅ API pagination complete for {domain_name} at page {page} (empty response)", "INFO")
                        break
                    
                    # Extract products from response
                    page_products = []
                    for item in data:
                        name = item.get("name", "").strip()
                        source_url = item.get("source_url", "").strip()
                        
                        if name and source_url:
                            page_products.append({
                                "productName": name,
                                "productLink": source_url,
                            })
                        else:
                            _log(f"⚠️ Skipping malformed product entry: name='{name}', url='{source_url}'", "WARNING")
                    
                    if not page_products:
                        _log(f"✅ API pagination complete for {domain_name} at page {page} (no valid products)", "INFO")
                        break
                    
                    products.extend(page_products)
                    _log(f"📦 Fetched {len(page_products)} products from page {page} (total: {len(products)})", "INFO")
                    
                    # If we got fewer than 500 products, we've likely reached the end
                    if len(page_products) < 500:
                        _log(f"✅ API pagination complete for {domain_name} (received {len(page_products)} < 500 products)", "INFO")
                        break
                    
                    page += 1
                    
                    # Add a small delay to be respectful to the API
                    await asyncio.sleep(0.5)
                    
            except asyncio.TimeoutError:
                _log(f"⏱️ Timeout fetching page {page} for {domain_name}", "ERROR")
                break
            except aiohttp.ClientError as e:
                _log(f"❌ Network error fetching page {page} for {domain_name}: {str(e)}", "ERROR")
                break
            except json.JSONDecodeError as e:
                _log(f"❌ Invalid JSON response for page {page} of {domain_name}: {str(e)}", "ERROR")
                break
            except Exception as e:
                _log(f"❌ Unexpected error fetching page {page} for {domain_name}: {str(e)}", "ERROR")
                break
    
    finally:
        if close_session:
            await session.close()
    
    _log(f"🎯 Total products fetched for {domain_name}: {len(products)}", "INFO")
    return products

# Global PDF tracker with asyncio-safe state management
class PDFStatusTracker:
    """
    Thread-safe PDF status tracker for managing deduplication and cleaning state.
    Uses asyncio primitives for safe concurrent access.
    """
    def __init__(self):
        self._pdf_status_map = {}  # {pdf_id: status_dict}
        self._lock = asyncio.Lock()  # Global lock for map modifications
    
    async def get_or_create_status(self, pdf_id: str, initial_path: str = None):
        """
        Get or create a status entry for a PDF.
        Returns the status dict with its condition variable.
        """
        async with self._lock:
            if pdf_id not in self._pdf_status_map:
                self._pdf_status_map[pdf_id] = {
                    'cleaned': False,
                    'cleaned_path': initial_path,
                    'original_path': initial_path,
                    'pending_copy_paths': [],
                    'condition': asyncio.Condition(),
                    'cleaning_in_progress': False,
                }
            return self._pdf_status_map[pdf_id]
    
    async def mark_cleaning_started(self, pdf_id: str):
        """Mark that cleaning has started for this PDF."""
        async with self._lock:
            if pdf_id in self._pdf_status_map:
                self._pdf_status_map[pdf_id]['cleaning_in_progress'] = True
    
    async def mark_cleaned(self, pdf_id: str, cleaned_path: str):
        """
        Mark a PDF as cleaned and notify all waiting tasks.
        Also process any pending copy operations.
        """
        status = await self.get_or_create_status(pdf_id)
        async with status['condition']:
            status['cleaned'] = True
            status['cleaned_path'] = cleaned_path
            status['cleaning_in_progress'] = False
            
            # Process pending copies
            for target_path in status['pending_copy_paths']:
                try:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.copy2(cleaned_path, target_path)
                    log_message(f"📋 Copied cleaned PDF to: {os.path.basename(target_path)}", "INFO")
                except Exception as e:
                    log_message(f"⚠️ Failed to copy cleaned PDF to {target_path}: {e}", "WARNING")
            
            # Clear pending copies
            status['pending_copy_paths'].clear()
            
            # Notify all waiting tasks
            status['condition'].notify_all()
    
    async def mark_cleaning_failed(self, pdf_id: str):
        """Mark that cleaning failed for this PDF and notify waiters."""
        status = await self.get_or_create_status(pdf_id)
        async with status['condition']:
            status['cleaning_in_progress'] = False
            # Notify waiters so they don't hang forever
            status['condition'].notify_all()
    
    async def wait_for_cleaned(self, pdf_id: str, target_path: str, timeout: float = 600):
        """
        Wait for a PDF to be cleaned, then copy it to target_path.
        Returns True if successful, False if timeout or failure.
        """
        status = await self.get_or_create_status(pdf_id)
        
        try:
            async with status['condition']:
                # If already cleaned, copy immediately
                if status['cleaned'] and status['cleaned_path']:
                    try:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        shutil.copy2(status['cleaned_path'], target_path)
                        log_message(f"📋 Copied already-cleaned PDF to: {os.path.basename(target_path)}", "INFO")
                        return True
                    except Exception as e:
                        log_message(f"⚠️ Failed to copy cleaned PDF: {e}", "WARNING")
                        return False
                
                # Add to pending copies and wait
                status['pending_copy_paths'].append(target_path)
                log_message(f"⏳ Waiting for PDF cleaning to complete (timeout: {timeout}s)...", "INFO")
                
                # Wait with timeout
                try:
                    await asyncio.wait_for(
                        status['condition'].wait_for(lambda: status['cleaned'] or not status['cleaning_in_progress']),
                        timeout=timeout
                    )
                    
                    # Check if cleaning succeeded
                    if status['cleaned']:
                        return True
                    else:
                        log_message(f"⚠️ PDF cleaning failed or was cancelled", "WARNING")
                        # Remove from pending since it won't be copied
                        if target_path in status['pending_copy_paths']:
                            status['pending_copy_paths'].remove(target_path)
                        return False
                        
                except asyncio.TimeoutError:
                    log_message(f"⏱️ Timeout waiting for PDF cleaning", "WARNING")
                    # Remove from pending
                    if target_path in status['pending_copy_paths']:
                        status['pending_copy_paths'].remove(target_path)
                    return False
                    
        except Exception as e:
            log_message(f"❌ Error waiting for cleaned PDF: {e}", "ERROR")
            return False
    
    async def cleanup_unfinished(self):
        """
        Clean up any PDFs that were never cleaned successfully.
        Call this at the end of category processing.
        """
        async with self._lock:
            for pdf_id, status in list(self._pdf_status_map.items()):
                if not status['cleaned'] and status['original_path']:
                    try:
                        if os.path.exists(status['original_path']):
                            log_message(f"🗑️ Removing uncleaned PDF: {os.path.basename(status['original_path'])}", "INFO")
                            # Note: We keep the original file for now, just log
                    except Exception as e:
                        log_message(f"⚠️ Error during cleanup: {e}", "WARNING")
                
                # Log any pending copies that never completed
                if status['pending_copy_paths']:
                    log_message(f"⚠️ PDF had {len(status['pending_copy_paths'])} pending copies that were never completed", "WARNING")
    
    async def get_stats(self):
        """Get statistics about tracked PDFs."""
        async with self._lock:
            total = len(self._pdf_status_map)
            cleaned = sum(1 for s in self._pdf_status_map.values() if s['cleaned'])
            in_progress = sum(1 for s in self._pdf_status_map.values() if s['cleaning_in_progress'])
            pending = sum(len(s['pending_copy_paths']) for s in self._pdf_status_map.values())
            return {
                'total_pdfs': total,
                'cleaned': cleaned,
                'in_progress': in_progress,
                'pending_copies': pending
            }

# Global instance
_pdf_tracker = PDFStatusTracker()

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

def generate_product_name_js_commands(
    primary_selector: str,
    click_selectors: list[str] | None = None,
    shadow_collect_selectors: list[str] | None = None,
) -> str:
    """
    Generate enhanced JavaScript commands for product name extraction.
    
    Args:
        primary_selector (str): The primary CSS selector from CSV configuration
        click_selectors (list[str] | None): Optional selectors to click before extraction
        
    Returns:
        str: JavaScript code for product name extraction
    """
    click_selectors = click_selectors or []
    shadow_collect_selectors = [s for s in (shadow_collect_selectors or []) if isinstance(s, str) and s.strip()]
    serialized_selectors = json.dumps(click_selectors, ensure_ascii=False)
    serialized_shadow_selectors = json.dumps(shadow_collect_selectors, ensure_ascii=False)
    return f"""


        // --- Shadow DOM helper (best-effort) ---
        // Many modern sites (e.g. Schneider Electric) render document lists inside open shadow roots.
        // document.querySelectorAll(...) does NOT pierce shadow boundaries, so we collect matching nodes
        // from open shadow roots and clone them into a hidden light-DOM container.
        try {{

            await new Promise(r => setTimeout(r, 3000));
            const __crawlerShadowSelectors = {serialized_shadow_selectors};
            const __crawlerShadowContainerId = "__crawler_shadow_container";
            const __crawlerMaxElementsVisited = 25000;
            const __crawlerMaxShadowHosts = 500;
            const __crawlerMaxClonedMatches = 4000;

            // Deep query helpers that traverse open shadow roots (for clicking + product name extraction).
            // NOTE: these can "see" into open shadow roots, but Crawl4AI target_elements selection may not,
            // hence we still clone matches into a light-DOM container for extraction.
            function __crawlerDeepQueryAll(selector) {{
                const results = [];
                if (!selector || !selector.trim()) return results;
                let elementsVisited = 0;
                let hostsVisited = 0;
                const queue = [];
                if (document.documentElement) queue.push(document.documentElement);
                while (queue.length) {{
                    const el = queue.shift();
                    if (!el || el.nodeType !== 1) continue;
                    elementsVisited++;
                    if (elementsVisited > __crawlerMaxElementsVisited) break;
                    // Search within this root if it supports querySelectorAll
                    try {{
                        if (el.querySelectorAll) {{
                            const found = el.querySelectorAll(selector);
                            if (found && found.length) {{
                                for (let i = 0; i < found.length; i++) results.push(found[i]);
                            }}
                        }}
                    }} catch (e) {{
                        // ignore invalid selectors / query errors
                    }}
                    // Traverse light DOM children
                    if (el.children && el.children.length) {{
                        for (let i = 0; i < el.children.length; i++) queue.push(el.children[i]);
                    }}
                    // Traverse open shadow root
                    const sr = el.shadowRoot;
                    if (sr) {{
                        hostsVisited++;
                        if (hostsVisited > __crawlerMaxShadowHosts) break;
                        try {{
                            const walker = document.createTreeWalker(sr, NodeFilter.SHOW_ELEMENT);
                            let node = walker.nextNode();
                            let shadowVisited = 0;
                            while (node) {{
                                queue.push(node);
                                shadowVisited++;
                                if (shadowVisited > 5000) break;
                                node = walker.nextNode();
                            }}
                        }} catch (e) {{
                            // ignore traversal failures
                        }}
                        // Also query directly inside shadow root
                        try {{
                            const foundInShadow = sr.querySelectorAll(selector);
                            if (foundInShadow && foundInShadow.length) {{
                                for (let i = 0; i < foundInShadow.length; i++) results.push(foundInShadow[i]);
                            }}
                        }} catch (e) {{
                            // ignore
                        }}
                    }}
                    if (results.length >= 2000) break; // cap to avoid runaway
                }}
                return results;
            }}

            function __crawlerDeepQueryFirst(selector) {{
                const all = __crawlerDeepQueryAll(selector);
                return all && all.length ? all[0] : null;
            }}

            function __crawlerEnsureShadowContainer() {{
                let c = document.getElementById(__crawlerShadowContainerId);
                if (!c) {{
                    c = document.createElement("div");
                    c.id = __crawlerShadowContainerId;
                    c.setAttribute("data-crawler", "shadow-container");
                    c.style.cssText = "display:none !important; visibility:hidden !important;";
                    (document.body || document.documentElement).appendChild(c);
                }}
                // Clear any previous run
                c.innerHTML = "";
                return c;
            }}

            function __crawlerCollectFromOpenShadowRoots() {{
                if (!__crawlerShadowSelectors || __crawlerShadowSelectors.length === 0) {{
                    return {{ hostsVisited: 0, matchesCloned: 0, elementsVisited: 0 }};
                }}

                const container = __crawlerEnsureShadowContainer();
                let elementsVisited = 0;
                let hostsVisited = 0;
                let matchesCloned = 0;

                // BFS across both light DOM and discovered shadow DOM trees
                const queue = [];
                if (document.documentElement) queue.push(document.documentElement);

                while (queue.length > 0) {{
                    const el = queue.shift();
                    if (!el || el.nodeType !== 1) continue;

                    elementsVisited++;
                    if (elementsVisited > __crawlerMaxElementsVisited) break;

                    // Traverse regular DOM children
                    if (el.children && el.children.length) {{
                        for (let i = 0; i < el.children.length; i++) {{
                            queue.push(el.children[i]);
                        }}
                    }}

                    // Traverse open shadow root (if present)
                    const sr = el.shadowRoot;
                    if (sr) {{
                        hostsVisited++;
                        if (hostsVisited > __crawlerMaxShadowHosts) break;

                        // Enqueue shadow DOM elements for nested shadow roots discovery
                        try {{
                            const walker = document.createTreeWalker(sr, NodeFilter.SHOW_ELEMENT);
                            let node = walker.nextNode();
                            let shadowVisited = 0;
                            while (node) {{
                                queue.push(node);
                                shadowVisited++;
                                if (shadowVisited > 5000) break; // cap per shadow root
                                node = walker.nextNode();
                            }}
                        }} catch (e) {{
                            // ignore traversal failures
                        }}

                        // Clone matches into light DOM so other extraction steps can see them
                        for (const sel of __crawlerShadowSelectors) {{
                            if (!sel || !sel.trim()) continue;
                            let matches = [];
                            try {{
                                matches = sr.querySelectorAll(sel);
                            }} catch (e) {{
                                continue;
                            }}
                            if (!matches || matches.length === 0) continue;

                            for (let i = 0; i < matches.length; i++) {{
                                if (matchesCloned >= __crawlerMaxClonedMatches) break;
                                const m = matches[i];
                                try {{
                                    const wrap = document.createElement("div");
                                    wrap.setAttribute("data-crawler-shadow-host", (el.tagName || "").toLowerCase());
                                    if (el.id) wrap.setAttribute("data-crawler-shadow-host-id", el.id);
                                    wrap.setAttribute("data-crawler-shadow-selector", sel);
                                    wrap.appendChild(m.cloneNode(true));
                                    container.appendChild(wrap);
                                    matchesCloned++;
                                }} catch (e) {{
                                    // ignore clone failures
                                }}
                            }}
                            if (matchesCloned >= __crawlerMaxClonedMatches) break;
                        }}
                    }}

                    if (matchesCloned >= __crawlerMaxClonedMatches) break;
                }}

                return {{ hostsVisited, matchesCloned, elementsVisited }};
            }}

            const __crawlerShadowStats = __crawlerCollectFromOpenShadowRoots();
            console.log("[JS] Shadow DOM collection:", JSON.stringify(__crawlerShadowStats));
        }} catch (error) {{
            console.log("[JS] Shadow DOM collection error:", error && error.message ? error.message : String(error));
        }}

        // ------------------------------------------------------------ mitsubishi electric --------------------------------------------------------
        
        try {{
                function findDownloadsElement(selectors) {{
                for (const selector of selectors) {{
                const elements = document.querySelectorAll(selector);
            
                for (const el of elements) {{
                    if (el.textContent.trim() === "DOWNLOADS") {{
                    return el; // Found the element
                    }}
                }}
                }}
            
                return null; // Not found
            }}
            
            // Example usage:
            await new Promise(r => setTimeout(r, 3000));
            const selectors = ["button.Nav_link__yt4y8.Tabs_navLink__SEkZx.Tabs_root--navLink__6yUNs"];
            const downloadsElement = findDownloadsElement(selectors);
            
            if (downloadsElement) {{
                console.log("Found:", downloadsElement);
                downloadsElement.click();
                await new Promise(r => setTimeout(r, 1000));
                document.querySelectorAll('span.Collapsible_icon__EIKfk.icon-plus')[0].click();
                await new Promise(r => setTimeout(r, 1000));
                document.querySelectorAll('span.Collapsible_icon__EIKfk.icon-plus')[0].click();
                await new Promise(r => setTimeout(r, 1000));
                document.querySelectorAll('span.Collapsible_icon__EIKfk.icon-plus')[0].click();
                await new Promise(r => setTimeout(r, 1000));
                document.querySelectorAll('span.Collapsible_icon__EIKfk.icon-plus')[0].click();
                await new Promise(r => setTimeout(r, 1000));
            }} else {{
                console.log("Element with text 'Downloads' not found.");
            }}
  
        }} catch (error) {{
            console.log('[JS] Error clicking language selector:', error.message);
        }}

        // ------------------------------------------------------------ mitsubishi electric --------------------------------------------------------

        const clickSelectors = {serialized_selectors};
        if (clickSelectors.length > 0) {{
            console.log('[JS] Pre-click selectors detected:', clickSelectors);
        }} else {{
            console.log('[JS] No pre-click selectors provided.');
        }}

        for (const cssSelector of clickSelectors) {{
            if (!cssSelector || !cssSelector.trim()) {{
                continue;
            }}
            try {{
                const button = __crawlerDeepQueryFirst(cssSelector) || document.querySelector(cssSelector);
                if (button) {{
                    console.log('[JS] Clicking pre-download selector:', cssSelector);
                    button.click();
                }} else {{
                    console.log('[JS] Pre-download selector not found:', cssSelector);
                }}
            }} catch (error) {{
                console.log('[JS] Error clicking selector', cssSelector, ':', error.message);
            }}
            await new Promise(r => setTimeout(r, 3000));
        }}

        console.log('[JS] Starting enhanced product name extraction...');
        await new Promise(r => setTimeout(r, 3000));
        
        // Special case: Try extracting from h1.childNodes[2] first (for specific site structure)
        var productName = "";
        var usedSelector = "";
        try {{
            var h1 = document.querySelector("h1");
            if (h1 && h1.childNodes && h1.childNodes.length > 2) {{
                var childNode = h1.childNodes[2];
                if (childNode && childNode.nodeType === 3) {{ // Text node
                    var text = childNode.textContent.trim().replace(/"/g, '');
                    if (text && text.length > 2 && text.length < 200) {{
                        productName = text;
                        usedSelector = "h1.childNodes[2]";
                        console.log('[JS] Successfully extracted product name from h1.childNodes[2]:', productName);
                        console.log('[JS] Used selector: h1.childNodes[2]');
                        return productName;
                    }}
                }}
            }}
        }} catch (error) {{
            console.log('[JS] Error with h1.childNodes[2] extraction:', error.message);
        }}

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
        
        var extractionAttempts = [];
        
        // Try each selector until we find a valid product name
        for (var i = 0; i < allSelectors.length; i++) {{
            try {{
                var selector = allSelectors[i];
                console.log('[JS] Trying selector:', selector);
                
                var element = __crawlerDeepQueryFirst(selector) || document.querySelector(selector);
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
        try:
            print(f"[{level}] {message}")
        except UnicodeEncodeError:
            # Fallback for Windows terminals that can't handle emojis
            ascii_message = message.encode('ascii', 'ignore').decode('ascii')
            print(f"[{level}] {ascii_message}")


def sanitize_folder_name(product_name: str) -> str:
    """
    Sanitizes a product name to be used as a folder name.
    Removes or replaces invalid characters that could cause folder creation issues.
    
    Args:
        product_name (str): The original product name
        
    Returns:
        str: Sanitized folder name safe for filesystem use
    """
    # Remove or replace invalid characters for folder names.
    # Windows invalid: < > : " | ? * \ /
    # Unix/Linux invalid: /
    #
    # Project convention (traceable): replace each special char with a distinct token.
    # Example: "\" -> "_#1_", "/" -> "_#2_", "<" -> "_#3_", etc.
    # This avoids different illegal chars collapsing into the same "_" and helps debugging.
    
    # First, replace newlines and tabs with spaces
    sanitized = product_name.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Replace special characters with deterministic tokens
    special_map = {
        "\\": "_#1_",
        "/": "_#2_",
        ":": "_#3_",
        "\"": "_#4_",
        "|": "_#5_",
        "?": "_#6_",
        "*": "_#7_",
    }
    for ch, repl in special_map.items():
        if ch in sanitized:
            sanitized = sanitized.replace(ch, repl)

    # Replace remaining ASCII control chars (safety; Windows disallows these in names)
    sanitized = re.sub(r'[\x00-\x1f]', ' ', sanitized)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Normalize whitespace (but do NOT collapse underscores; tokens like "_#1_" rely on them)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
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
        'application/x-edz': 'edz',
        'application/edz': 'edz',
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


def normalize_language_code(language: str) -> str:
    """
    Normalize language strings to short uppercase codes where possible.
    Examples: "English" -> "EN", "DE", "TR"; otherwise "Unknown".
    """
    if not language:
        return "Unknown"
    lang = language.strip().lower()
    if lang in {"en", "eng", "english"}:
        return "EN"
    if lang in {"de", "ger", "deu", "german", "deutsch"}:
        return "DE"
    if lang in {"tr", "tur", "turkish", "türkçe"}:
        return "TR"
    # Common others
    if lang in {"fr", "fra", "fre", "french"}:
        return "FR"
    if lang in {"es", "spa", "spanish"}:
        return "ES"
    return "Unknown"


def parse_content_disposition_filename(header_value: str) -> str:
    """
    Extract filename from Content-Disposition header (RFC 5987/6266 handling).
    Returns empty string if none.
    """
    if not header_value:
        return ""
    try:
        value = header_value
        # filename* takes precedence
        match_star = re.search(r"filename\*=([^']*)''([^;]+)", value, re.IGNORECASE)
        if match_star:
            # filename*=utf-8''encoded
            enc = match_star.group(1) or 'utf-8'
            raw = match_star.group(2)
            try:
                from urllib.parse import unquote
                decoded = unquote(raw)
                return decoded
            except Exception:
                return raw
        match = re.search(r'filename="?([^";]+)"?', value, re.IGNORECASE)
        if match:
            return match.group(1)
    except Exception:
        return ""
    return ""


def sniff_extension_from_bytes(content: bytes) -> str:
    """
    Infer file extension by inspecting magic bytes/content.
    Handles PDF, ZIP, STEP/STP, IGES, DWG, DXF, STL (basic heuristics).
    Returns empty string if unknown.
    """
    if not content:
        return ""
    head = content[:512]
    # PDF
    if head.startswith(b"%PDF"):
        return "pdf"
    # ZIP (also common for many CAD package downloads)
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06") or head.startswith(b"PK\x07\x08"):
        return "zip"
    # STEP/STP
    if b"ISO-10303-21" in head or b"STEP-File" in head:
        return "step"
    # IGES: often starts with 'S      1' in ASCII fixed-width records
    if head[:1] == b'S' and b"IGES" in head:
        return "iges"
    # DWG: AC10xx signature
    if head.startswith(b"AC1"):
        return "dwg"
    # DXF: ASCII begins with 0\nSECTION or 999\n
    try:
        txt = head.decode(errors='ignore')
        if txt.lstrip().startswith("0\nSECTION") or txt.lstrip().startswith("999\n") or txt.lstrip().upper().startswith("SECTION"):
            return "dxf"
        # STL ASCII: starts with 'solid'
        if txt.lstrip().lower().startswith("solid"):
            return "stl"
    except Exception:
        pass
    # STL binary: 80-byte header then uint32 triangles; heuristic: if size fits pattern
    if len(content) > 84:
        try:
            import struct
            tri_count = struct.unpack('<I', content[80:84])[0]
            # Each triangle is 50 bytes; check rough size consistency (allow slack)
            if 80 + 4 + tri_count * 50 <= len(content) + 200:
                return "stl"
        except Exception:
            pass
    return ""


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


def _process_pdf_wrapper(file_path: str, api_key: str) -> dict:
    """
    Wrapper function for PDF processing to be called in a separate process.
    This function must be at module level for pickling by ProcessPoolExecutor.
    
    Args:
        file_path: Path to the PDF file to process
        api_key: API key for AI processing
        
    Returns:
        dict: Result with 'success', 'file_path', and optional 'error' keys
    """
    try:
        # Define a simple log callback that doesn't use external state
        # Using ASCII characters to avoid encoding issues in subprocess
        def log_callback(message, level="INFO"):
            try:
                # Try to print with UTF-8, fallback to ASCII if it fails
                print(f"[{level}] [{os.path.basename(file_path)}] {message}", flush=True)
            except UnicodeEncodeError:
                # Fallback: remove non-ASCII characters
                ascii_message = message.encode('ascii', 'ignore').decode('ascii')
                print(f"[{level}] [{os.path.basename(file_path)}] {ascii_message}", flush=True)
        
        pdf_processing(file_path=file_path, api_key=api_key, log_callback=log_callback)
        return {
            'success': True,
            'file_path': file_path
        }
    except Exception as e:
        return {
            'success': False,
            'file_path': file_path,
            'error': str(e)
        }


def _cleanup_failed_pdf(file_path: str, log_message_func=None) -> None:
    """
    Clean up a failed PDF file and its temporary artifacts.
    
    Args:
        file_path: Path to the PDF file to remove
        log_message_func: Optional logging function
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            if log_message_func:
                log_message_func(f"🗑️ Removed failed PDF: {os.path.basename(file_path)}", "INFO")
    except Exception as e:
        if log_message_func:
            log_message_func(f"⚠️ Failed to remove file {os.path.basename(file_path)}: {str(e)}", "WARNING")

def build_cookie_header(cookie_dict: dict[str, str]) -> str: 
    return "; ".join(f"{k}={v}" for k,v in cookie_dict.items())

async def get_browser_cookies(crawler: AsyncWebCrawler) -> dict[str, str]: 
    context = await crawler.browser_manger.get__primary_context()
    cookies = await context.cookies()

    return {cookie["name"]: cookie["value"] for cookie in cookies}


async def download_pdfs_via_playwright(
    product_url: str,
    pdf_button_selector: str,
    click_selectors: list[str],
    output_folder: str,
    api_key: str,
    cat_name: str = "Uncategorized",
    preserve_product_name: str = None,
) -> tuple:
    """
    Handle browser-triggered downloads using Playwright download events.
    This is for websites where clicking a link triggers a browser "Save As" dialog
    instead of providing a direct downloadable URL.
    
    Args:
        product_url: URL of the product page
        pdf_button_selector: CSS selector for download links/buttons
        output_folder: Base output directory for downloads
        api_key: API key for potential future use
        cat_name: Category name for folder organization
        preserve_product_name: If provided, use this name instead of extracting from HTML (for API mode)
        
    Returns:
        Tuple of (list of file paths, product name, category path, product path)
        Returns ([], "Unknown Product", "", "") on failure
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        log_message("❌ Playwright not installed. Run: pip install playwright && python -m playwright install", "ERROR")
        return ([], "Unknown Product", "", "")
    
    log_message(f"🎭 Using Playwright to handle browser-triggered downloads", "INFO")
    log_message(f"🌐 Product URL: {product_url}", "INFO")
    log_message(f"🔘 Button selector: {pdf_button_selector}", "INFO")
    
    downloaded_files = []
    derived_product_name = "Unknown Product"
    category_path = ""
    product_path = ""
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                # Headless is more reliable for automation, and also enables Chromium PDF generation
                # (used to bypass native "Save As" dialogs triggered by print-preview flows).
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-features=VizDisplayCompositor',
                ]
            )
            
            # Enable downloads in context
            context = await browser.new_context(
                accept_downloads=True,
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )

            # Some sites "download" via window.print() which opens browser print preview + a native save dialog.
            # We can't automate native dialogs, so we intercept window.print() and generate a PDF via Playwright.
            try:
                await context.add_init_script("""
                    (() => {
                        try {
                            window.__crawlerPrintCalled = false;
                            const originalPrint = window.print;
                            window.print = () => {
                                window.__crawlerPrintCalled = true;
                                // do NOT open print preview
                            };
                            window.__crawlerOriginalPrint = originalPrint;
                        } catch (e) {
                            // ignore
                        }
                    })();
                """)
            except Exception:
                # If init script injection fails, the rest of the pipeline still works for normal downloads.
                pass
            
            page = await context.new_page()
            
            # Track downloads info
            downloads_info = []
            
            async def handle_download(download):
                """Handle each download event"""
                try:
                    filename = download.suggested_filename
                    log_message(f"📥 Download triggered: {filename}", "INFO")
                    
                    # Generate safe filename
                    safe_filename = sanitize_filename(filename)
                    
                    # Create temporary path (will be moved later)
                    temp_path = os.path.join(output_folder, safe_filename)
                    os.makedirs(output_folder, exist_ok=True)
                    
                    # Save the download
                    await download.save_as(temp_path)
                    
                    file_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                    downloads_info.append({
                        'url': download.url,
                        'filename': filename,
                        'path': temp_path,
                        'size': file_size,
                        'method': 'download_event'
                    })
                    
                    log_message(f"✅ Saved: {safe_filename} ({file_size / 1024:.1f} KB)", "INFO")
                    
                except Exception as e:
                    log_message(f"❌ Error saving download '{filename}': {str(e)}", "ERROR")
            
            # Register download handler
            page.on("download", handle_download)
            
            # Navigate to product page
            log_message(f"🌐 Loading product page...", "INFO")
            try:
                await page.goto(product_url, wait_until='networkidle', timeout=60000)
            except Exception as e:
                log_message(f"⚠️ Page load warning: {str(e)}", "WARNING")
                # Try to continue anyway
                await page.wait_for_timeout(3000)
            
            # Wait for page to settle
            await page.wait_for_timeout(2000)
            
            # Try to dismiss cookie consent banners and overlays
            try:
                log_message(f"🍪 Attempting to dismiss cookie consent banners...", "INFO")
                # Common cookie consent selectors
                cookie_selectors = [
                    'button:has-text("Accept")',
                    'button:has-text("Accept all")',
                    'button:has-text("Agree")',
                    'button:has-text("OK")',
                    'button:has-text("I agree")',
                    'button[id*="accept"]',
                    'button[class*="accept"]',
                    'button[class*="consent"]',
                    '.privacy_prompt button',
                    '#__tealiumGDPRecModal button',
                    'a:has-text("Accept all cookies")',
                ]
                
                for selector in cookie_selectors:
                    try:
                        elements = await page.locator(selector).all()
                        for element in elements:
                            if await element.is_visible():
                                await element.click(timeout=2000)
                                log_message(f"✅ Dismissed cookie banner with selector: {selector}", "INFO")
                                await page.wait_for_timeout(1000)
                                break
                    except Exception:
                        continue
            except Exception as e:
                log_message(f"⚠️ Could not dismiss cookie banner: {str(e)}", "WARNING")

            # Optional pre-clicks (passed from CSV) to reveal downloads or switch tabs before scraping download buttons.
            # We intentionally do this AFTER cookie dismissal and BEFORE product name extraction / download element lookup.
            normalized_click_selectors = [s.strip() for s in (click_selectors or []) if isinstance(s, str) and s.strip()]
            if normalized_click_selectors:
                log_message(f"🖱️ Running {len(normalized_click_selectors)} pre-click selector(s) before download discovery", "INFO")
                known_pages = set(context.pages)
                for sel_idx, sel in enumerate(normalized_click_selectors, 1):
                    try:
                        locator = page.locator(sel)
                        count = await locator.count()
                        if count <= 0:
                            log_message(f"⏭️ Pre-click selector not found ({sel_idx}/{len(normalized_click_selectors)}): {sel}", "INFO")
                            await page.wait_for_timeout(800)
                            continue

                        clicked = False
                        # If selector matches multiple elements, only click the one whose innerText is "FILES".
                        # This avoids accidentally clicking unrelated tab buttons with the same selector.
                        if count > 1:
                            target_idx = None
                            for i in range(min(count, 25)):
                                el = locator.nth(i)
                                try:
                                    if not await el.is_visible():
                                        continue
                                    txt = await el.inner_text()
                                    if (txt or "").strip().lower() == "files":
                                        target_idx = i
                                        break
                                except Exception:
                                    continue

                            if target_idx is None:
                                log_message(
                                    f"⏭️ Pre-click selector matched {count} elements but none had innerText='FILES' ({sel_idx}/{len(normalized_click_selectors)}): {sel}",
                                    "INFO",
                                )
                                await page.wait_for_timeout(800)
                                continue

                            try:
                                el = locator.nth(target_idx)
                                await el.scroll_into_view_if_needed(timeout=3000)
                                await el.click(timeout=8000)
                                clicked = True
                                log_message(
                                    f"✅ Pre-clicked ({sel_idx}/{len(normalized_click_selectors)}) by text='FILES': {sel}",
                                    "INFO",
                                )
                            except Exception as e:
                                log_message(
                                    f"⚠️ Pre-click failed on element with innerText='FILES' ({sel_idx}/{len(normalized_click_selectors)}): {sel} ({e})",
                                    "WARNING",
                                )
                                await page.wait_for_timeout(800)
                                continue
                        else:
                            # Single match: click if visible.
                            el = locator.first
                            try:
                                if await el.is_visible():
                                    await el.scroll_into_view_if_needed(timeout=3000)
                                    await el.click(timeout=8000)
                                    clicked = True
                                    log_message(f"✅ Pre-clicked ({sel_idx}/{len(normalized_click_selectors)}): {sel}", "INFO")
                                else:
                                    log_message(f"⏭️ Pre-click element not visible ({sel_idx}/{len(normalized_click_selectors)}): {sel}", "INFO")
                                    await page.wait_for_timeout(800)
                            except Exception:
                                pass

                        if not clicked:
                            # Last resort: force click first match (can help for offscreen overlays / custom controls).
                            try:
                                await locator.first.click(timeout=8000, force=True)
                                clicked = True
                                log_message(f"✅ Pre-clicked (forced) ({sel_idx}/{len(normalized_click_selectors)}): {sel}", "INFO")
                            except Exception as force_err:
                                log_message(f"⚠️ Pre-click failed ({sel_idx}/{len(normalized_click_selectors)}): {sel} ({force_err})", "WARNING")

                        # If the click opened a new tab, close it (we don't want to hijack the main flow here).
                        await page.wait_for_timeout(1500)
                        try:
                            for p2 in list(context.pages):
                                if p2 is page:
                                    continue
                                if p2 not in known_pages:
                                    try:
                                        log_message(f"🧹 Closing unexpected tab opened during pre-click: {p2.url}", "INFO")
                                    except Exception:
                                        log_message("🧹 Closing unexpected tab opened during pre-click", "INFO")
                                    try:
                                        await p2.close()
                                    except Exception:
                                        pass
                            known_pages = set(context.pages)
                        except Exception:
                            pass
                    except Exception as e:
                        log_message(f"⚠️ Error during pre-click ({sel_idx}/{len(normalized_click_selectors)}): {sel} ({str(e)})", "WARNING")
                        await page.wait_for_timeout(800)
            
            # Extract product name using JavaScript (or use preserved name from API)
            if preserve_product_name:
                derived_product_name = preserve_product_name.strip()
                log_message(f"📝 Using preserved product name from API: {derived_product_name}", "INFO")
            else:
                try:
                    # Reuse the product name extraction logic - wrap in arrow function for Playwright
                    derived_product_name = await page.evaluate("""
                        () => {
                            let productName = "";
                            const selectors = ['h1', '.product-title', '.product-name', 'h1.title', 'h2'];
                            for (let sel of selectors) {
                                const el = document.querySelector(sel);
                                if (el && el.innerText && el.innerText.trim().length > 2) {
                                    productName = el.innerText.trim();
                                    break;
                                }
                            }
                            return productName || "Unknown Product";
                        }
                    """)
                    if derived_product_name and derived_product_name != "Unknown Product":
                        log_message(f"📝 Product name extracted: {derived_product_name}", "INFO")
                except Exception as e:
                    log_message(f"⚠️ Could not extract product name: {str(e)}", "WARNING")
                    derived_product_name = "Unknown Product"
            
            # Create folder structure
            sanitized_cat_name = sanitize_folder_name(cat_name)
            category_path = os.path.join(output_folder, sanitized_cat_name)
            os.makedirs(category_path, exist_ok=True)
            
            product_path = os.path.join(category_path, sanitize_folder_name(derived_product_name))
            os.makedirs(product_path, exist_ok=True)
            log_message(f"📁 Created folder structure: {sanitized_cat_name}/{sanitize_folder_name(derived_product_name)}", "INFO")
            
            # Find all download buttons/links
            try:
                elements = await page.locator(pdf_button_selector).all()
                log_message(f"🔍 Found {len(elements)} download element(s)", "INFO")
                
                if not elements:
                    log_message(f"⚠️ No elements found for selector: {pdf_button_selector}", "WARNING")
                    await browser.close()
                    return ([], derived_product_name, category_path, product_path)
                
                # Track statistics for logging
                download_event_count = 0
                new_tab_count = 0
                failed_count = 0
                
                # Random delay before starting downloads (be polite to servers)
                import random
                pre_download_delay = random.uniform(3, 8)
                log_message(f"⏳ Waiting {pre_download_delay:.1f}s before starting downloads...", "INFO")
                await page.wait_for_timeout(int(pre_download_delay * 1000))
                
                # Click each element and wait for downloads OR new tabs
                for i, element in enumerate(elements, 1):
                    new_tab = None
                    try:
                        # Check if element is visible and enabled
                        try:
                            is_visible = await element.is_visible()
                            is_enabled = await element.is_enabled()
                            
                            if not is_visible:
                                log_message(f"⏭️ Button {i}/{len(elements)} is not visible, skipping", "INFO")
                                failed_count += 1
                                continue
                            
                            if not is_enabled:
                                log_message(f"⏭️ Button {i}/{len(elements)} is disabled, skipping", "INFO")
                                failed_count += 1
                                continue
                        except Exception as visibility_err:
                            log_message(f"⚠️ Could not check button {i} visibility: {str(visibility_err)}", "WARNING")
                        
                        # Get element text for logging
                        try:
                            text = await element.inner_text()
                            text_preview = text.strip()[:50] if text else "No text"
                        except:
                            text_preview = "Unable to read text"
                        
                        log_message(f"🖱️ Clicking element {i}/{len(elements)}: {text_preview}", "INFO")
                        
                        # Dual-listener approach: listen for BOTH download event AND new tab
                        download_triggered = False
                        new_tab_opened = False
                        pdf_url_from_tab = None
                        
                        try:
                            # Create tasks for both events
                            async def wait_for_new_page():
                                """Wait for a new page/tab to open"""
                                try:
                                    new_page = await context.wait_for_event('page', timeout=15000)
                                    return ('new_tab', new_page)
                                except Exception:
                                    return ('new_tab', None)
                            
                            async def wait_for_download():
                                """Wait for a download event"""
                                try:
                                    async with page.expect_download(timeout=15000) as download_info:
                                        return ('download', await download_info.value)
                                except Exception:
                                    return ('download', None)
                            
                            # Start both listeners
                            new_page_task = asyncio.create_task(wait_for_new_page())
                            download_task = asyncio.create_task(wait_for_download())
                            
                            # Click the button
                            await element.click()
                            
                            # Wait a bit for events to fire
                            await page.wait_for_timeout(1500)
                            
                            # Check which events completed
                            done, pending = await asyncio.wait(
                                [new_page_task, download_task],
                                timeout=16.0,
                                return_when=asyncio.ALL_COMPLETED
                            )
                            
                            # Process completed tasks
                            for task in done:
                                try:
                                    event_type, result = await task
                                    
                                    if event_type == 'new_tab' and result:
                                        new_tab = result
                                        new_tab_opened = True
                                        # Wait for the new tab to load
                                        try:
                                            await new_tab.wait_for_load_state('load', timeout=10000)
                                            pdf_url_from_tab = new_tab.url
                                            log_message(f"📑 Button {i} opened new tab with URL: {pdf_url_from_tab}", "INFO")
                                        except Exception as load_err:
                                            log_message(f"⚠️ New tab load timeout: {str(load_err)}", "WARNING")
                                            pdf_url_from_tab = new_tab.url if new_tab else None
                                    
                                    elif event_type == 'download' and result:
                                        download_triggered = True
                                        log_message(f"📥 Button {i} triggered download event", "INFO")
                                        # Download will be handled by the registered handler
                                
                                except Exception as task_err:
                                    log_message(f"⚠️ Error processing task result: {str(task_err)}", "WARNING")
                            
                            # Cancel any pending tasks
                            for task in pending:
                                task.cancel()
                            
                            # Handle the new tab scenario - download PDF from URL
                            # Only process if no download was triggered (prefer download event over new tab)
                            if new_tab_opened and pdf_url_from_tab and not download_triggered:
                                # Validate URL before attempting download
                                if pdf_url_from_tab and len(pdf_url_from_tab) > 5 and ('http://' in pdf_url_from_tab or 'https://' in pdf_url_from_tab):
                                    new_tab_count += 1
                                    try:
                                        # Close the tab immediately to free resources
                                        await new_tab.close()
                                        
                                        # Download the PDF manually using Playwright's request API
                                        log_message(f"⬇️ Downloading PDF from new tab URL: {pdf_url_from_tab}", "INFO")
                                        
                                        response = await context.request.get(pdf_url_from_tab, timeout=60000)
                                        
                                        if response.ok:
                                            content = await response.body()
                                            
                                            # Determine filename from URL or Content-Disposition header
                                            filename = None
                                            content_disposition = response.headers.get('content-disposition', '')
                                            if content_disposition and 'filename=' in content_disposition:
                                                import re
                                                match = re.search(r'filename[*]?=["\']?([^"\';\r\n]+)', content_disposition)
                                                if match:
                                                    filename = match.group(1)
                                            
                                            if not filename:
                                                # Extract from URL
                                                from urllib.parse import urlparse, unquote
                                                parsed = urlparse(pdf_url_from_tab)
                                                filename = unquote(os.path.basename(parsed.path))
                                                if not filename or '.' not in filename:
                                                    filename = f"document_{i}.pdf"
                                            
                                            # Sanitize filename
                                            safe_filename = sanitize_filename(filename)
                                            temp_path = os.path.join(output_folder, safe_filename)
                                            os.makedirs(output_folder, exist_ok=True)
                                            
                                            # Write file
                                            with open(temp_path, 'wb') as f:
                                                f.write(content)
                                            
                                            file_size = len(content)
                                            downloads_info.append({
                                                'url': pdf_url_from_tab,
                                                'filename': filename,
                                                'path': temp_path,
                                                'size': file_size,
                                                'method': 'new_tab'
                                            })
                                            
                                            log_message(f"✅ Downloaded from new tab: {safe_filename} ({file_size / 1024:.1f} KB)", "SUCCESS")
                                        else:
                                            log_message(f"❌ Failed to download from new tab URL: HTTP {response.status}", "ERROR")
                                            failed_count += 1
                                    
                                    except Exception as tab_err:
                                        log_message(f"❌ Error handling new tab download: {str(tab_err)}", "ERROR")
                                        failed_count += 1
                                else:
                                    # Print-preview / internal pages (e.g. chrome://print) can't be fetched via request.get().
                                    # As a fallback, render the page to a PDF using Chromium's PDF generator.
                                    try:
                                        new_tab_count += 1
                                        try:
                                            await new_tab.wait_for_load_state('domcontentloaded', timeout=8000)
                                        except Exception:
                                            pass

                                        base_name = derived_product_name if derived_product_name and derived_product_name != "Unknown Product" else f"document_{i}"
                                        safe_pdf_name = sanitize_filename(f"{base_name}.pdf")
                                        temp_path = os.path.join(output_folder, safe_pdf_name)
                                        os.makedirs(output_folder, exist_ok=True)

                                        log_message(f"🖨️ Detected non-HTTP(S) tab ({pdf_url_from_tab}); saving via page.pdf(): {safe_pdf_name}", "INFO")
                                        try:
                                            await new_tab.wait_for_load_state('networkidle', timeout=8000)
                                        except Exception:
                                            pass
                                        await new_tab.pdf(
                                            path=temp_path,
                                            print_background=True,
                                            prefer_css_page_size=True,
                                        )

                                        file_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                                        downloads_info.append({
                                            'url': pdf_url_from_tab or '',
                                            'filename': safe_pdf_name,
                                            'path': temp_path,
                                            'size': file_size,
                                            'method': 'print_to_pdf_tab'
                                        })
                                        log_message(f"✅ Saved via page.pdf(): {safe_pdf_name} ({file_size / 1024:.1f} KB)", "SUCCESS")
                                    except Exception as pdf_err:
                                        log_message(f"⚠️ New tab URL is invalid and PDF rendering failed: {str(pdf_err)} (url={pdf_url_from_tab})", "WARNING")
                                        failed_count += 1
                            
                            if download_triggered:
                                download_event_count += 1
                                log_message(f"✅ Button {i} successfully triggered download event", "INFO")
                            
                            else:
                                # Neither download nor new tab: maybe the site called window.print().
                                try:
                                    print_called = await page.evaluate("() => Boolean(window.__crawlerPrintCalled)")
                                except Exception:
                                    print_called = False

                                if print_called:
                                    try:
                                        base_name = derived_product_name if derived_product_name and derived_product_name != "Unknown Product" else f"document_{i}"
                                        safe_pdf_name = sanitize_filename(f"{base_name}.pdf")
                                        temp_path = os.path.join(output_folder, safe_pdf_name)
                                        os.makedirs(output_folder, exist_ok=True)

                                        log_message(f"🖨️ Detected window.print() call; saving current page via page.pdf(): {safe_pdf_name}", "INFO")
                                        try:
                                            await page.wait_for_load_state('networkidle', timeout=8000)
                                        except Exception:
                                            pass
                                        await page.pdf(
                                            path=temp_path,
                                            print_background=True,
                                            prefer_css_page_size=True,
                                        )

                                        file_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                                        downloads_info.append({
                                            'url': page.url,
                                            'filename': safe_pdf_name,
                                            'path': temp_path,
                                            'size': file_size,
                                            'method': 'print_to_pdf_page'
                                        })
                                        log_message(f"✅ Saved via page.pdf(): {safe_pdf_name} ({file_size / 1024:.1f} KB)", "SUCCESS")
                                    except Exception as pdf_err:
                                        log_message(f"⚠️ window.print() detected but PDF rendering failed: {str(pdf_err)}", "WARNING")
                                        failed_count += 1
                                else:
                                    log_message(f"⚠️ Button {i} did not trigger download, open new tab, or call window.print()", "WARNING")
                                    failed_count += 1
                        
                        except Exception as click_err:
                            log_message(f"⚠️ Error during button {i} click handling: {str(click_err)}", "WARNING")
                            failed_count += 1
                            continue
                        
                    except Exception as e:
                        log_message(f"⚠️ Error processing element {i}: {str(e)}", "WARNING")
                        failed_count += 1
                        continue
                    
                    finally:
                        # Ensure new tab is closed even if error occurred
                        if new_tab:
                            try:
                                await new_tab.close()
                            except:
                                pass
                        
                        # Random delay between downloads (be polite to servers)
                        if i < len(elements):
                            between_delay = random.uniform(2, 5)
                            log_message(f"⏳ Waiting {between_delay:.1f}s before next download...", "INFO")
                            await page.wait_for_timeout(int(between_delay * 1000))
                
                # Wait for all downloads to complete
                await page.wait_for_timeout(2000)
                
                # Log statistics
                log_message(f"📊 Button click summary: {download_event_count} download events, {new_tab_count} new tabs, {failed_count} failed", "INFO")
                
            except Exception as e:
                log_message(f"❌ Error finding/clicking elements: {str(e)}", "ERROR")
            
            await browser.close()
            
            # Move downloaded files to proper product folder
            for download_info in downloads_info:
                try:
                    temp_path = download_info['path']
                    filename = os.path.basename(temp_path)
                    final_path = os.path.join(product_path, filename)
                    
                    if os.path.exists(temp_path):
                        shutil.move(temp_path, final_path)
                        downloaded_files.append(final_path)
                        log_message(f"📦 Moved to: {os.path.relpath(final_path, output_folder)}", "INFO")
                except Exception as e:
                    log_message(f"⚠️ Failed to move file: {str(e)}", "WARNING")
            
            if downloaded_files:
                log_message(f"✅ Successfully downloaded {len(downloaded_files)} file(s) via Playwright", "SUCCESS")
            else:
                log_message(f"⚠️ No files were downloaded", "WARNING")
            
            return (downloaded_files, derived_product_name, category_path, product_path)
            
    except Exception as e:
        log_message(f"❌ Playwright download error: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        return ([], derived_product_name, category_path, product_path)


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
        pdf_button_selector: str = "",
        click_selectors: list[str] | None = None,
        preserve_product_name: str = None,
        ):
    
    """
    Opens the given product page and uses CSS selector to target PDF links.
    Uses LLMExtractionStrategy to identify and extract technical PDF documents,
    then downloads them. Prevents downloading duplicate PDFs by checking existing files.
    Creates a category-based folder structure: output/{category}/{product}
    
    Args:
        crawler: The AsyncWebCrawler instance
        product_url: URL of the product page
        output_folder: Base output directory for downloads
        pdf_llm_strategy: LLM extraction strategy for PDF processing
        pdf_selector: CSS selector to target PDF link elements
        session_id: Session identifier for crawling
        regex_strategy: Optional regex extraction strategy (unused)
        domain_name: Domain name for URL completion
        api_key: API key for LLM provider
        cat_name: Category name from CSV for folder organization (defaults to "Uncategorized")
        pdf_button_selector: CSS selector for PDF download buttons
        click_selectors: List of CSS selectors to click before extraction via JS commands
        preserve_product_name: If provided, use this name instead of extracting from HTML (for API mode)
    """


    click_selectors = [selector.strip() for selector in (click_selectors or []) if selector and selector.strip()]


    # Global dictionary to track downloaded files with their file paths across all products
    if not hasattr(download_pdf_links, 'downloaded_files'):
        download_pdf_links.downloaded_files = {}  # Dict: {url: file_path} for all file types

    # Early return: If pdf_button_selector provided, use Playwright directly (skip LLM extraction)
    if pdf_button_selector and pdf_button_selector.strip():
        log_message(f"🎭 Detected pdf_button_selector {{{pdf_button_selector}}}, using Playwright for button-triggered downloads", "INFO")
        try:
            # Check if Playwright is available
            try:
                from playwright.async_api import async_playwright
            except ImportError:
                log_message(f"❌ Playwright required but not installed. Run: pip install playwright", "ERROR")
                return {"productLink": product_url, "productName": None, "category": cat_name, "saved_count": 0, "has_datasheet": False}
            
            product_url = product_url.replace("?tab=detail", "?tab=downloads")
            # Execute Playwright download
            playwright_files, derived_product_name, category_path, product_path = await download_pdfs_via_playwright(
                product_url=product_url,
                pdf_button_selector=pdf_button_selector,
                click_selectors=click_selectors,
                output_folder=output_folder,
                api_key=api_key,
                cat_name=cat_name,
                preserve_product_name=preserve_product_name,
            )
            
            if not playwright_files:
                log_message(f"⚠️ Playwright found no downloads for: {product_url}", "WARNING")
                return {"productLink": product_url, "productName": derived_product_name, "category": cat_name, "saved_count": 0, "has_datasheet": False}
            
            log_message(f"📥 Playwright downloaded {len(playwright_files)} file(s)", "INFO")
            
            # Separate PDFs from other files for cleaning
            pdfs_to_clean = []
            saved_files = []
            has_datasheet = False
            
            for file_path in playwright_files:
                saved_files.append(file_path)
                # Check if it's a PDF
                if file_path.lower().endswith('.pdf'):
                    pdfs_to_clean.append((file_path, product_url))
                    # Check if it's a datasheet
                    basename = os.path.basename(file_path).lower()
                    if 'datasheet' in basename or 'data_sheet' in basename or 'specification' in basename:
                        has_datasheet = True
            
            # Parallel PDF cleaning (reuse existing pipeline)
            if pdfs_to_clean:
                log_message(f"🧹 Starting parallel PDF cleaning for {len(pdfs_to_clean)} PDF(s)", "INFO")
                
                # Determine worker count
                cpu_count = multiprocessing.cpu_count()
                max_workers = max(1, min(cpu_count // 1, 2))
                log_message(f"🔧 Using {max_workers} parallel workers for PDF cleaning", "INFO")
                
                # Timeout per PDF: 10 minutes (600 seconds)
                PDF_TIMEOUT = 600
                
                # Track successful and failed PDFs
                successful_pdfs = []
                failed_pdfs = []
                
                # Process PDFs in parallel using ProcessPoolExecutor
                loop = asyncio.get_event_loop()
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all PDF processing tasks
                    future_to_pdf = {}
                    for pdf_info in pdfs_to_clean:
                        pdf_path, pdf_url = pdf_info
                        future = executor.submit(_process_pdf_wrapper, pdf_path, api_key)
                        future_to_pdf[future] = (pdf_path, pdf_url)
                    
                    # Wait for results with timeout handling
                    for future, (pdf_path, pdf_url) in future_to_pdf.items():
                        pdf_name = os.path.basename(pdf_path)
                        try:
                            successful_pdfs.append(pdf_path)
                            log_message(f"✨ PDF cleaning completed: {pdf_name}", "INFO")
                            # Notify tracker that PDF is cleaned
                            pdf_id = pdf_url
                            await _pdf_tracker.mark_cleaned(pdf_id, pdf_path)
                        
                        except FuturesTimeoutError:
                            failed_pdfs.append(pdf_path)
                            log_message(f"⏱️ PDF cleaning timeout ({PDF_TIMEOUT}s) for {pdf_name}", "ERROR")
                            log_message(f"🗑️ Removing hung PDF: {pdf_name}", "WARNING")
                            _cleanup_failed_pdf(pdf_path, log_message)
                            # Notify tracker of failure
                            pdf_id = pdf_url
                            await _pdf_tracker.mark_cleaning_failed(pdf_id)
                            # Remove from saved_files list
                            if pdf_path in saved_files:
                                saved_files.remove(pdf_path)
                        
                        except Exception as e:
                            failed_pdfs.append(pdf_path)
                            log_message(f"❌ Unexpected error cleaning {pdf_name}: {str(e)}", "ERROR")
                            _cleanup_failed_pdf(pdf_path, log_message)
                            # Notify tracker of failure
                            pdf_id = pdf_url
                            await _pdf_tracker.mark_cleaning_failed(pdf_id)
                            # Remove from saved_files list
                            if pdf_path in saved_files:
                                saved_files.remove(pdf_path)
                
                # Summary of PDF cleaning
                log_message(f"📊 PDF cleaning summary: {len(successful_pdfs)} successful, {len(failed_pdfs)} failed", "INFO")
                if failed_pdfs:
                    log_message(f"❌ Failed PDFs: {', '.join([os.path.basename(p) for p in failed_pdfs])}", "WARNING")
            
            # Return summary
            return {
                "productLink": product_url,
                "productName": derived_product_name,
                "category": cat_name,
                "saved_count": len(saved_files),
                "has_datasheet": has_datasheet,
            }
            
        except Exception as e:
            log_message(f"❌ Playwright download failed for {product_url}: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
            return {"productLink": product_url, "productName": None, "category": cat_name, "saved_count": 0, "has_datasheet": False}

    try:
        log_message(f"🔍 Starting file extraction for product page", "INFO")
        log_message(f"📍 Using selectors: {pdf_selector}", "INFO")

        # Normalize selectors: callers typically pass a list, but some entrypoints may pass a string.
        if isinstance(pdf_selector, list):
            normalized_selectors = [str(s).strip() for s in pdf_selector if str(s).strip()]
        else:
            normalized_selectors = [s.strip() for s in str(pdf_selector or "").split("|") if s.strip()]
        if not normalized_selectors:
            log_message("⚠️ No valid pdf_selector provided; skipping PDF extraction", "WARNING")
            return
        pdf_selector = normalized_selectors

        log_message(f"Name Selector: {pdf_selector[-1]}", "INFO")


        # Enhanced JavaScript product name extraction using helper function
        # The second selector will be used as primary, with comprehensive fallback selectors
        product_url = f"{product_url}"
        js_commands = generate_product_name_js_commands(
            pdf_selector[-1],
            click_selectors=click_selectors,
            shadow_collect_selectors=pdf_selector,
        )
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
        # If preserve_product_name is provided (from API mode), use it directly
        if preserve_product_name:
            derived_product_name = preserve_product_name.strip()
            log_message(f"📝 Using preserved product name from API: '{derived_product_name}'", "INFO")
        else:
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

        # Create the download folder structure after product name extraction
        os.makedirs(output_folder, exist_ok=True)
        sanitized_cat_name = sanitize_folder_name(cat_name)
        category_path = os.path.join(output_folder, sanitized_cat_name)
        os.makedirs(category_path, exist_ok=True)
        
        # productPath = os.path.join(category_path, sanitize_folder_name(derived_product_name))
        # if not os.path.exists(productPath):
        #     os.makedirs(productPath)
        #     log_message(f"📁 Created folder structure: {sanitized_cat_name}/{sanitize_folder_name(derived_product_name)}", "INFO")

        # Step 3: Split extracted files into PDFs and other file types
        # and apply per-type limits with strong language preference (EN first).
        pdf_docs = []
        other_docs = []
        seen_urls_in_page = set()
        pdf_cfg = DEFAULT_CONFIG.get("pdf_settings", {})
        allowed_types = set(t.lower() for t in pdf_cfg.get("allowed_types", [])) if pdf_cfg else set()
        per_type_limits = {k.lower(): v for k, v in (pdf_cfg.get("per_type_limits", {}) or {}).items()}
        disallowed_types = set(t.lower() for t in pdf_cfg.get("disallowed_types", [])) if pdf_cfg else set()
        per_type_counts: dict[str, int] = {}
        # Collect all candidates first so we can re-order within each type by language & priority
        candidates: list[dict] = []

        # with open("extracted_data.json", "w", encoding="utf-8") as f:
        #     json.dump(extracted_data, f, ensure_ascii=False, indent=4)

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
            
            # Allowed types filtering (by semantic type string from LLM)
            item_type = (item.get('type') or 'Unknown').strip()
            item_type_lc = item_type.lower()
            if (allowed_types and item_type_lc not in allowed_types) or (disallowed_types and item_type_lc in disallowed_types):
                log_message(f"⏭️ Skipping disallowed type: {item_type}", "INFO")
                continue

            # Store as candidate; we'll apply per-type limits *after* we sort by language/priority
            candidates.append({
                "item": item,
                "ext": ext,
                "type_lc": item_type_lc,
                "type_label": item_type,
            })

        # Enhanced sorting: Priority first, then by document type preference, then by language
        def sort_key(doc: dict):
            """
            Sorting key applied to the LLM-extracted document dict.
            Priority (High/Medium/Low) > document type (Data Sheet > Drawing > Manual > CAD > others) > language (EN > DE > TR > others).
            """
            # Priority first
            priority_score = {'High': 3, 'Medium': 2, 'Low': 1}.get(doc.get('priority', 'Unknown'), 0)
            
            # Type preference: Data Sheet > Technical Drawing > Manual > CAD > Others
            type_score = 0
            doc_type = (doc.get('type') or '').lower()
            if 'data sheet' in doc_type or 'datasheet' in doc_type or 'specification' in doc_type:
                type_score = 4
            elif 'drawing' in doc_type or 'dimensional' in doc_type:
                type_score = 3
            elif 'manual' in doc_type or 'guide' in doc_type:
                type_score = 2
            elif 'cad' in doc_type or 'step' in doc_type or 'iges' in doc_type or 'dwg' in doc_type:
                type_score = 1
            
            # Language preference: English > German > Turkish > Others (based on normalized code)
            lang_code = normalize_language_code(doc.get('language', 'Unknown'))
            language_score_map = {
                "EN": 4,
                "DE": 3,
                "TR": 2,
            }
            language_score = language_score_map.get(lang_code, 1)
            
            return (priority_score, type_score, language_score)

        # Apply per-type limits with language-aware ordering
        if candidates:
            # Group candidates by semantic type
            from collections import defaultdict
            grouped_by_type: dict[str, list[dict]] = defaultdict(list)
            for c in candidates:
                grouped_by_type[c["type_lc"]].append(c)

            for type_lc, docs in grouped_by_type.items():
                item_type_label = docs[0]["type_label"] if docs else type_lc
                limit = per_type_limits.get(type_lc)

                # Sort documents of this type using the enhanced sort_key on the underlying LLM doc
                docs.sort(key=lambda c: sort_key(c["item"]), reverse=True)

                # Enforce per-type limit after sorting so that preferred language/docs are kept
                count = 0
                for c in docs:
                    if limit is not None and count >= limit:
                        log_message(f"📊 Skipping '{item_type_label}' (per-type limit {limit} reached with language-aware ordering)", "INFO")
                        continue

                    item = c["item"]
                    ext = c["ext"]

                    if is_pdf_extension(ext):
                        pdf_docs.append(item)
                    else:
                        other_docs.append(item)

                    per_type_counts[type_lc] = per_type_counts.get(type_lc, 0) + 1
                    count += 1

                    file_type = item.get('type', 'Unknown')
                    file_text = item.get('text', 'Unknown')
                    lang = item.get('language', 'Unknown')
                    if is_pdf_extension(ext):
                        log_message(f"✅ Accepted PDF: {file_type} ({lang}) - {file_text}", "INFO")
                    else:
                        log_message(f"✅ Accepted {ext.upper() if ext else 'file'}: {file_type} ({lang}) - {file_text}", "INFO")

        # Check if any files were found
        total_files = len(pdf_docs) + len(other_docs)
        if total_files == 0:
            log_message(f"📭 No files found on page for product: {derived_product_name}", "INFO")
            log_message(f"🔗 Product URL: {product_url}", "INFO")
            return {"productLink": product_url, "productName": derived_product_name, "category": cat_name, "saved_count": 0, "has_datasheet": False}
        else:
            # create a folder structure for the product
            sanitized_cat_name = sanitize_folder_name(cat_name)
            category_path = os.path.join(output_folder, sanitized_cat_name)
            os.makedirs(category_path, exist_ok=True)
            productPath = os.path.join(category_path, sanitize_folder_name(derived_product_name))
            if not os.path.exists(productPath):
                os.makedirs(productPath)
                log_message(f"📁 Created folder structure: {sanitized_cat_name}/{sanitize_folder_name(derived_product_name)}", "INFO")

        log_message(f"📄 Found {len(pdf_docs)} PDF(s) and {len(other_docs)} other file(s) for product: {derived_product_name}", "INFO")

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

        bm = crawler.crawler_strategy.browser_manager
        pdf_session_id = f"{session_id}_pdf_extraction"   # the same session_id you pass to CrawlerRunConfig

        # after crawler.arun(...) completes
        context, page, _ = bm.sessions[pdf_session_id]    # stored as (context, page, last_used_ts)

        cookies = await context.cookies()
        cookie_header = "; ".join(f"{c['name']}={c['value']}" for c in cookies)
        ua = crawler.browser_config.user_agent


        request_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": product_url,
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
            "Cookie": cookie_header,
        }

        saved_files = []
        pdfs_to_clean = []  # Collect PDFs for parallel processing
        has_datasheet = False
        timeout = aiohttp.ClientTimeout(total=90, connect=15, sock_read=60)
       
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=True), 
            timeout=timeout,
            headers=request_headers,
        ) as session: # try to add headers to the request 
            
            for i, file_info in enumerate(all_files, 1):
                file_url = file_info['url']
                file_text = file_info['text']
                file_type = file_info['type']
                file_language = normalize_language_code(file_info.get('language', 'Unknown'))
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
                            # Always use allowed_types from config for filename base
                            filename = make_config_type_filename(file_type, file_language, file_ext or 'bin', productPath)
                            save_path = os.path.join(productPath, filename)
                            
                            # Ensure product folder exists before copying (in case it wasn't created earlier)
                            folder_created = not os.path.exists(productPath)
                            os.makedirs(productPath, exist_ok=True)
                            if folder_created:
                                log_message(f"📁 Created folder structure: {sanitized_cat_name}/{sanitize_folder_name(derived_product_name)}", "INFO")
                            # Note: Folder should already exist from line 1383, this is a safety check
                            
                            # For PDFs, check if cleaned before copying
                            if is_pdf:
                                pdf_id = file_url  # Use URL as unique identifier
                                status = await _pdf_tracker.get_or_create_status(pdf_id, previous_path)
                                
                                # If already cleaned, copy immediately
                                if status['cleaned'] and status['cleaned_path']:
                                    shutil.copy2(status['cleaned_path'], save_path)
                                    log_message(f"📋 Copied previously cleaned PDF: {filename}", "INFO")
                                    saved_files.append(save_path)
                                    if file_type and ('data sheet' in file_type.lower() or 'datasheet' in file_type.lower() or 'specification' in file_type.lower()):
                                        has_datasheet = True
                                    continue
                                else:
                                    # PDF not yet cleaned, wait for it
                                    log_message(f"⏳ PDF exists but not cleaned yet, will wait: {filename}", "INFO")
                                    wait_success = await _pdf_tracker.wait_for_cleaned(pdf_id, save_path, timeout=600)
                                    if wait_success:
                                        saved_files.append(save_path)
                                        if file_type and ('data sheet' in file_type.lower() or 'datasheet' in file_type.lower() or 'specification' in file_type.lower()):
                                            has_datasheet = True
                                        continue
                                    else:
                                        log_message(f"⚠️ Failed to get cleaned PDF, will re-download: {file_url}", "WARNING")
                            else:
                                # Non-PDF files, copy directly
                                shutil.copy2(previous_path, save_path)
                                log_message(f"📋 Copied previously downloaded {file_type_label}: {filename}", "INFO")
                                saved_files.append(save_path)
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
                    # basic retry with backoff
                    attempt = 0
                    max_attempts = 3
                    last_exc = None
                    while attempt < max_attempts:
                        try:
                            resp = await session.get(
                                file_url,
                                headers={**request_headers, "Referer": product_url, "User-Agent": request_headers["User-Agent"]},
                                allow_redirects=True
                            )
                            if resp.status == 403: 
                                print("403 error, retrying...")
                                await asyncio.sleep(1.5)
                                resp = await session.get(
                                    file_url,
                                    headers={**request_headers, "Referer": product_url, "User-Agent": request_headers["User-Agent"]},
                                    allow_redirects=True,
                                )
                            break
                        except Exception as req_exc:
                            last_exc = req_exc
                            attempt += 1
                            await asyncio.sleep(1.5 * attempt)
                    if attempt == max_attempts and last_exc:
                        raise last_exc
                    async with resp:
                        if resp.status == 200:
                            # Refine extension from response headers
                            refined_ext = infer_extension_from_url_and_headers(file_url, resp.headers)
                            if refined_ext:
                                file_ext = refined_ext
                                is_pdf = is_pdf_extension(file_ext)
                            # Attempt filename from Content-Disposition
                            cd_header = resp.headers.get('content-disposition') or resp.headers.get('Content-Disposition')
                            cd_filename = parse_content_disposition_filename(cd_header) if cd_header else ""
                            
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
                            
                            # Validate and sniff extension from bytes
                            sniffed_ext = sniff_extension_from_bytes(content)
                            if sniffed_ext:
                                file_ext = sniffed_ext
                                is_pdf = is_pdf_extension(file_ext)
                            
                            # Validate content: must be either PDF or a recognized CAD/document format
                            if not is_pdf and len(content) >= 4 and not content.startswith(b'%PDF'):
                                # Allow other recognized file types (DWG, STEP, IGES, DXF, STL, ZIP, etc.)
                                allowed_extensions = ['dwg', 'dxf', 'step', 'stp', 'iges', 'igs', 'stl', 'zip', 'edz']
                                if file_ext.lower() not in allowed_extensions:
                                    log_message(f"⚠️ Skipping unrecognized file type '{file_ext}' from {file_url}", "INFO")
                                    continue
                                else:
                                    log_message(f"✅ Accepted non-PDF file type: {file_ext.upper()}", "INFO")
                            
                            # Check if content is identical to any existing file
                            content_hash = hashlib.md5(content).hexdigest()
                            
                            # Store content hash for deduplication
                            if not hasattr(download_pdf_links, 'content_hashes'):
                                download_pdf_links.content_hashes = set()
                            download_pdf_links.content_hashes.add(content_hash)
                            
                            # Always use allowed_types from config for filename base
                            filename = make_config_type_filename(file_type, file_language, file_ext or 'bin', productPath)

                            # Filename is already unique and final from make_config_type_filename
                            save_path = os.path.join(productPath, filename)
                            
                            # Save the file
                            async with aiofiles.open(save_path, "wb") as f:
                                await f.write(content)


                            # Post-download certificate filter (only for PDFs)
                            if is_pdf:
                                lowered = os.path.basename(save_path).lower()
                                cert_terms = [
                                    'certificate','certification','certifications','iso','tse', 'declaration', 'Garanti', 'Garanti Belgisi', 'Generic - Garanti Belgesi'
                                    'coc','iec','emc','fda','rohs','iecex','csa','warranty','contacts','list of ifm contacts','list of contacts'
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

                            # Collect PDFs for parallel processing later
                            if is_pdf:
                                pdfs_to_clean.append((save_path, file_url))  # Store both path and URL
                                # Register PDF with tracker and mark as cleaning in progress
                                pdf_id = file_url
                                await _pdf_tracker.get_or_create_status(pdf_id, save_path)
                                await _pdf_tracker.mark_cleaning_started(pdf_id)
                                log_message(f"✅ Downloaded PDF (queued for cleaning): {os.path.basename(save_path)}", "INFO")
                            else:
                                log_message(f"✅ Saved {file_type_label}: {os.path.basename(save_path)}", "INFO")
                        else:
                            log_message(f"❌ Failed to download: {file_url} (Status: {resp.status})", "ERROR")
                except Exception as e:
                    log_message(f"❌ Error downloading {file_url}: {str(e)}", "ERROR")
        
        # Parallel PDF cleaning with timeout and error handling
        if pdfs_to_clean:
            log_message(f"🧹 Starting parallel PDF cleaning for {len(pdfs_to_clean)} PDF(s)", "INFO")
            
            # Determine worker count: use CPU count / 2, but at least 1 and at most 4
            cpu_count = multiprocessing.cpu_count()
            max_workers = max(1, min(cpu_count // 1, 2))
            # max_workers = 1
            log_message(f"🔧 Using {max_workers} parallel workers for PDF cleaning", "INFO")
            
            # Timeout per PDF: 5 minutes (300 seconds)
            PDF_TIMEOUT = 600
            
            # Track successful and failed PDFs
            successful_pdfs = []
            failed_pdfs = []
            
            # Process PDFs in parallel using ProcessPoolExecutor
            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all PDF processing tasks
                future_to_pdf = {}
                for pdf_info in pdfs_to_clean:
                    if isinstance(pdf_info, tuple):
                        pdf_path, pdf_url = pdf_info
                    else:
                        # Backwards compatibility
                        pdf_path = pdf_info
                        pdf_url = pdf_path
                    future = executor.submit(_process_pdf_wrapper, pdf_path, api_key)
                    future_to_pdf[future] = (pdf_path, pdf_url)
                
                # Wait for results with timeout handling
                for future, (pdf_path, pdf_url) in future_to_pdf.items():
                    pdf_name = os.path.basename(pdf_path)
                    try:

                        successful_pdfs.append(pdf_path)
                        log_message(f"✨ PDF cleaning completed: {pdf_name}", "INFO")
                        # Notify tracker that PDF is cleaned
                        pdf_id = pdf_url
                        await _pdf_tracker.mark_cleaned(pdf_id, pdf_path)


                        # # Wait for the result with timeout
                        # result = await loop.run_in_executor(
                        #     None, 
                        #     partial(future.result, timeout=PDF_TIMEOUT)
                        # )
                        
                        # if result['success']:
                        #     successful_pdfs.append(pdf_path)
                        #     log_message(f"✨ PDF cleaning completed: {pdf_name}", "INFO")
                        #     # Notify tracker that PDF is cleaned
                        #     pdf_id = pdf_url
                        #     await _pdf_tracker.mark_cleaned(pdf_id, pdf_path)
                        # else:
                        #     failed_pdfs.append(pdf_path)
                        #     error_msg = result.get('error', 'Unknown error')
                        #     log_message(f"❌ PDF cleaning failed for {pdf_name}: {error_msg}", "ERROR")
                        #     _cleanup_failed_pdf(pdf_path, log_message)
                        #     # Notify tracker of failure
                        #     pdf_id = pdf_url
                        #     await _pdf_tracker.mark_cleaning_failed(pdf_id)
                        #     # Remove from saved_files list
                        #     if pdf_path in saved_files:
                        #         saved_files.remove(pdf_path)
                    
                    except FuturesTimeoutError:
                        failed_pdfs.append(pdf_path)
                        log_message(f"⏱️ PDF cleaning timeout ({PDF_TIMEOUT}s) for {pdf_name}", "ERROR")
                        log_message(f"🗑️ Removing hung PDF: {pdf_name}", "WARNING")
                        _cleanup_failed_pdf(pdf_path, log_message)
                        # Notify tracker of failure
                        pdf_id = pdf_url
                        await _pdf_tracker.mark_cleaning_failed(pdf_id)
                        # Remove from saved_files list
                        if pdf_path in saved_files:
                            saved_files.remove(pdf_path)
                    
                    except Exception as e:
                        failed_pdfs.append(pdf_path)
                        log_message(f"❌ Unexpected error cleaning {pdf_name}: {str(e)}", "ERROR")
                        _cleanup_failed_pdf(pdf_path, log_message)
                        # Notify tracker of failure
                        pdf_id = pdf_url
                        await _pdf_tracker.mark_cleaning_failed(pdf_id)
                        # Remove from saved_files list
                        if pdf_path in saved_files:
                            saved_files.remove(pdf_path)
            
            # Summary of PDF cleaning
            log_message(f"📊 PDF cleaning summary: {len(successful_pdfs)} successful, {len(failed_pdfs)} failed", "INFO")
            if failed_pdfs:
                log_message(f"❌ Failed PDFs: {', '.join([os.path.basename(p) for p in failed_pdfs])}", "WARNING")
    
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
            'page': ['p', 'pg', 'page_num', 'page_number', 'pageNumber', 'currentPage','page'],
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
                if (i > 0 and path_parts[i-1].lower() in ['page', 'p', 'pg', 'products', 'category', 'catalog', 'currentPage']) or \
                   (i == len(path_parts) - 1 and len(path_parts) > 1):
                    log_message(f"📄 Found path-based page number: {part} in URL path", "INFO")
                    return int(part)
        
        # If no path-based pagination found, check query parameters
        pagination_params_to_check = [
            'page', 'p', 'pg', 'page_num', 'page_number', 'pageNumber',
            'offset', 'start', 'skip', 'from',
            'limit', 'size', 'per_page', 'items_per_page',
            'cursor', 'after', 'before', 'next', 'prev',
            'page_id', 'pageid', 'pageno', 'pagenum', 'currentPage'
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
                if (i > 0 and path_parts[i-1].lower() in ['page', 'p', 'pg','currentPage', 'products', 'category', 'catalog']) or \
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
                'page_id', 'pageid', 'pageno', 'pagenum', 'currentPage'
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
                    if (i > 0 and path_parts[i-1].lower() in ['page', 'p', 'pg', 'currentPage', 'products', 'category', 'catalog']) or \
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
                'currentPage', 'p', 'pg', 'page_num', 'page_number', 'pageNumber',
                'offset', 'start', 'skip', 'from',
                'limit', 'size', 'per_page', 'items_per_page',
                'cursor', 'after', 'before', 'next', 'prev',
                'page_id', 'pageid', 'pageno', 'pagenum','page'
                ]
            
            # Find and remove existing pagination parameter
            existing_pagination_param = None
            for param in pagination_params_to_remove:
                if param in query_params:
                    existing_pagination_param = param
                    query_params.pop(param, None)
                    break
            
            # Calculate pagination values based on type
            if pagination_type in [ "page","currentPage","p", "pg", "page_num", "page_number", "pageNumber","page"]:
                query_params[str(pagination_type)] = [str(page_number)]
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
                query_params[str(pagination_type)] = [str(page_number)]
            
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
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"  # Generate a random user agent string
    # print(repr(f"User agent: {user_agent}"))
    browser_type = "chromium"

    # https://docs.crawl4ai.com/core/browser-crawler-config/
    return BrowserConfig(
        browser_type=browser_type,  # Type of browser to simulate
        headless=True,  # Whether to run in headless mode (no GUI)
        viewport_height=1080,
        viewport_width=1920,
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
    # Allowed and disallowed types filtering
    pdf_cfg = DEFAULT_CONFIG.get("pdf_settings", {})
    allowed_types = set(t.lower() for t in pdf_cfg.get("allowed_types", [])) if pdf_cfg else set()
    disallowed_types = set(t.lower() for t in pdf_cfg.get("disallowed_types", [])) if pdf_cfg else set()

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
            "You are given filtered HTML from a product page.\n"
            " and anchors for downloadable technical documents and CAD files.\n"
            "Extract all technical PDFs, especially: Data Sheet, Installation Guide, Technical Drawing, Catalog, User Manual, and info cards.\n"
            "Your output MUST INCLUDE Installation Guides or instructions if present, as well as Data Sheets, Technical Drawings, User Manuals, Mounting Instructions, and any kind of setup, commissioning, or installation document, even if the label is a variant (e.g., 'Install Guide', 'Setup Manual', 'Mounting Guide', 'Assembly Instructions', 'Commissioning Guide', 'Start-up Guide', etc).\n"
            "The high priority files are: English Data Sheets, Installation Guides, Technical Drawings, User Manuals, Mounting Instructions, and any kind of setup, commissioning, or installation document.\n"
            "For each document, output: url, text, type, language, priority.\n"
            "- url: absolute link to the file. Convert relative links using the page domain.\n"
            "    - Do not add any escape characters to the url.\n"
            "- text: the link text, button text describing the file or file name.\n"
            "- type: one of {allowed_types}.\n"
            "    - For any file labeled with install/installing/setup/commissioning/mounting/start/assembly, use type 'Installation Guide' if the document's main focus is installation or mounting.\n"
            "    - If any file could match both Installation Guide and User Manual, prefer Installation Guide as the type.\n"
            " - disallowed_types: one of {disallowed_types}.\n"
            "    - If the file type is in the disallowed_types, do not extract it.\n"
            "- language: language code like EN/DE/TR or Unknown.\n"
            "- priority: High for Data Sheet, Installation Guide, User Manual, Technical Drawing, and CAD; Medium for Catalog, ZIP.\n"
            "   - 'Installation Guide' documents are always High priority.\n"
            "Do not extract a list of contact files.\n"
            "Ignore certificates and Garanti Belgisi/compliance-only links. Return a JSON array matching the schema."
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
        try{{

            await new Promise(r => setTimeout(r, 3000));
            var btn2 = document.querySelector("button[id='didomi-notice-agree-button']");
            if (btn2) {{
                btn2.click();
                console.log('[JS] Button clicked');
                await new Promise(r => setTimeout(r, 3000));
            }}
        }} catch (error) {{
            console.log('[JS] Error with button extraction:', error.message);
        }}   
        await new Promise(r => setTimeout(r, 5000));

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

    # with open("extracted_data.json", "w", encoding="utf-8") as f:
    #     json.dump(extracted_data, f, ensure_ascii=False, indent=4)
    
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
    JS-based extraction for sites with dynamic pagination (including \"Load more\" buttons).
    Extracts product rows client-side using JS, then applies LLM extraction per page.
    Returns (venues, no_results) just like fetch_and_process_page.
    """

    # Read JS safety limits from configuration / environment with sensible fallbacks
    max_js_pages = int(os.getenv("MAX_JS_PAGINATION_PAGES", DEFAULT_CONFIG.get("crawler_settings", {}).get("max_js_pagination_pages", 500)))
    max_js_clicks = int(os.getenv("MAX_JS_PAGINATION_CLICKS", DEFAULT_CONFIG.get("crawler_settings", {}).get("max_js_pagination_clicks", 100)))

    js_commands = f"""

        try{{

            await new Promise(r => setTimeout(r, 3000));
            var btn2 = document.querySelector("button[id='didomi-notice-agree-button']");
            if (btn2) {{
                btn2.click();
                console.log('[JS] Button clicked');
                await new Promise(r => setTimeout(r, 3000));
            }}
        }} catch (error) {{
            console.log('[JS] Error with button extraction:', error.message);
        }}   


        console.log('[JS] Starting data extraction...');
        let allRowsData = [];
        const rowSelectors = '{", ".join(elements)}';
        const buttonSelector = '{button_selector}';
        const maxPages = {max_js_pages};
        let currentPage = 1;
        let totalClicks = 0;
        const maxClicks = {max_js_clicks}; // safety cap in JS layer (also configurable in Python)

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

        // Always extract first page before any clicking
        allRowsData.push({{
            page: currentPage,
            data: extractRows()
        }});
        console.log(`[JS] Extracted initial page with ${{allRowsData[0].data.length}} rows`);

        await new Promise(r => setTimeout(r, 3000));
        // Only attempt pagination if valid button selector exists
        if (buttonSelector && buttonSelector.trim() !== '') {{
            console.log('[JS] Pagination / load-more detected. Starting automatic pagination...');
            let nextButton = document.querySelector(buttonSelector);

            // Track last known product count to detect when no new products are added
            let lastTotalCount = allRowsData[0].data.length;

            while (
                currentPage < maxPages &&
                totalClicks < maxClicks &&
                nextButton &&
                nextButton.offsetParent !== null &&
                !nextButton.disabled
            ) {{
                // Click to load next page or more products
                nextButton.click();
                totalClicks += 1;
                console.log(`[JS] Clicked pagination button #${{totalClicks}} (logical page ${{currentPage}})`);
                currentPage++;

                // Wait for new content to load
                await new Promise(r => setTimeout(r, 9000));

                // Extract snapshot after click
                const newPageData = extractRows();
                const newTotalCount = newPageData.length;

                allRowsData.push({{
                    page: currentPage,
                    data: newPageData
                }});
                console.log(`[JS] Snapshot after click #${{totalClicks}} (logical page ${{currentPage}}) with ${{newPageData.length}} rows`);

                // Stopping condition for load-more: no increase in total rows
                if (newTotalCount <= lastTotalCount) {{
                    console.log('[JS] No new products detected (row count did not increase). Stopping pagination.');
                    break;
                }}

                // Update counters and button reference after DOM changes
                lastTotalCount = newTotalCount;
                nextButton = document.querySelector(buttonSelector);
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

