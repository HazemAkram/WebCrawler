"""
DeepSeek AI Web Crawler
Copyright (c) 2025 Ayaz Mensyoğlu

This file is part of the DeepSeek AI Web Crawler project.
Licensed under the Apache License, Version 2.0.
See NOTICE file for additional terms and conditions.
"""

import asyncio
import csv
import random
import os
import sys
import gc
import aiohttp
import json

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv
from urllib.parse import urlparse

from utils.scraper_utils import download_pdf_links, set_log_callback as set_scraper_log_callback, _pdf_tracker
from config import REQUIRED_KEYS
from utils.scraper_utils import (
    fetch_and_process_page,
    get_browser_config,
    get_llm_strategy,
    get_pdf_llm_strategy,
    append_page_param,  
    get_regex_strategy,
    fetch_and_process_page_with_js,
    get_page_number,
    detect_pagination_type,
    sanitize_folder_name,
    fetch_products_from_api,
    get_browser_cookies_for_domain,
    create_unified_browser_context,
    close_unified_browser_context,
    fetch_products_from_api_via_browser
)

load_dotenv()

# Global logging function that can be set by the web interface
log_callback = None

def set_log_callback(callback):
    """Set the logging callback function for web interface integration"""
    global log_callback
    log_callback = callback
    # Also set the callback for scraper_utils module
    set_scraper_log_callback(callback)

def log_message(message, level="INFO"):
    """Log a message, either to console or web interface"""
    global log_callback
    if log_callback:
        log_callback(message, level)
    else:
        print(f"[{level}] {message}")

def read_sites_from_csv(input_file):
    log_message(f"🔄 Reading sites from CSV: {input_file}", "INFO")
    sites = []
    with open(input_file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        count = 2
        for row in reader:
            print(f"Row number {count} is being processed")
            count += 1
            
            # Check if this is API-based (domain_name) or HTML-based (url) input
            domain_name = row.get('domain_name', '').strip()
            url = row.get('url', '').strip()
            
            # Determine crawling mode
            if domain_name and not url:
                # API-based mode: fetch products from API
                mode = "api"
                log_message(f"   📡 Row {count-1}: API mode detected for domain '{domain_name}'", "INFO")
            elif url:
                # HTML-based mode: traditional crawling
                mode = "html"
                log_message(f"   🌐 Row {count-1}: HTML mode detected for URL '{url}'", "INFO")
            else:
                log_message(f"   ⚠️ Row {count-1}: Skipping - neither domain_name nor url provided", "WARNING")
                continue
            
            # Parse selectors (common to both modes)
            pdf_list = [s.strip() for s in row.get('pdf_selector', '').split('|') if s.strip()]
            pdf_button_selector = row.get('pdf_button_selector', '').strip()
            
            # ---- NEW (API PAGE RANGE) ----
            page_number_raw = row.get('page_number', '').strip()
            # Accept either a single page or a page range (e.g., '1' or '1-5')
            if page_number_raw:
                if '-' in page_number_raw:
                    start_page, end_page = map(int, page_number_raw.split('-', 1))
                else:
                    start_page = int(page_number_raw)
                    end_page = None
            else:
                start_page = 1
                end_page = None
            # Build site config based on mode
            if mode == "api":
                # Check if browser cookie extraction should be used (default: True)
                use_browser_cookies = row.get("use_browser_cookies", "true").strip().lower()
                use_browser_cookies = use_browser_cookies in ["true", "1", "yes", "y", ""]
                
                sites.append({
                    "mode": "api",
                    "domain_name": domain_name,
                    "cat_name": row.get("cat_name", domain_name),
                    "pdf_selector": pdf_list,
                    "pdf_button_selector": pdf_button_selector,
                    "start_page": start_page,
                    "end_page": end_page,
                    "use_browser_cookies": use_browser_cookies,
                })
            else:  # html mode
                css_list = [s.strip() for s in row.get('css_selector', '').split('|') if s.strip()]
                name_selector = row.get('name_selector', '').strip()
                if name_selector:
                    pdf_list.append(name_selector)
                
                sites.append({
                    "mode": "html",
                    "url": url,
                    "cat_name": row.get("cat_name", "Uncategorized"),
                    "css_selector": css_list,
                    "pdf_selector": pdf_list,
                    "button_selector": row.get("button_selector", ""),
                    "pdf_button_selector": pdf_button_selector,
                })
    return sites

async def cleanup_crawler_session(crawler, session_id):
    """
    Clean up crawler session to prevent resource leaks
    
    Args:
        crawler: AsyncWebCrawler instance
        session_id: Session identifier to clean up
    """
    try:
        # Force garbage collection
        gc.collect()
        log_message(f"🧹 Cleaned up session: {session_id}", "INFO")
    except Exception as e:
        log_message(f"⚠️ Session cleanup warning: {str(e)}", "WARNING")

async def process_single_product(
    venue: dict,
    site: dict,
    browser_config,
    pdf_llm_strategy,
    regex_strategy,
    api_key: str,
    session_id: str,
    domain_name: str,
    semaphore: asyncio.Semaphore,
    stop_requested_callback=None,
    page_number: int = None,
    cookies: dict = None,
    headers: dict = None,
    preserve_product_name: str = None,
):
    """
    Process a single product: download and clean PDFs.
    Uses a semaphore to limit concurrency and avoid overwhelming servers.
    Now returns detailed error information for reporting.
    
    Args:
        venue: Product information dict
        site: Site configuration dict
        browser_config: Browser configuration
        pdf_llm_strategy: PDF LLM strategy
        regex_strategy: Regex strategy
        api_key: API key for services
        session_id: Base session ID
        domain_name: Domain name
        semaphore: Asyncio semaphore for rate limiting
        stop_requested_callback: Callback to check if stop is requested
        page_number: API page number (for error reporting)
        
    Returns:
        Tuple of (summary_dict or None, error_dict or None)
    """
    async with semaphore:  # Control concurrency
        if stop_requested_callback and stop_requested_callback():
            log_message("Stop requested by user.", "WARNING")
            return None, None
        
        product_url = venue.get('productLink', '')
        product_name = venue.get('productName', 'Unknown')
        
        # First, check if product page is accessible
        try:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=True), cookies=cookies, headers=headers) as check_session:
                async with check_session.head(product_url, timeout=aiohttp.ClientTimeout(total=10), allow_redirects=True) as response:
                    if response.status >= 400:
                        error_info = {
                            "page_number": page_number,
                            "cat_name": site.get("cat_name", "Unknown"),
                            "productName": product_name,
                            "productLink": product_url,
                            "error_type": "product_page_access",
                            "http_status": response.status,
                            "error_message": f"HTTP {response.status}: {response.reason}"
                        }
                        log_message(f"❌ Product page inaccessible: {product_url} (HTTP {response.status})", "ERROR")
                        # return None, error_info
        except asyncio.TimeoutError:
            error_info = {
                "page_number": page_number,
                "cat_name": site.get("cat_name", "Unknown"),
                "productName": product_name,
                "productLink": product_url,
                "error_type": "product_page_timeout",
                "http_status": "N/A",
                "error_message": "Timeout checking product page accessibility"
            }
            log_message(f"⏱️ Timeout checking product page: {product_url}", "ERROR")
            return None, error_info
        except Exception as e:
            error_info = {
                "page_number": page_number,
                "cat_name": site.get("cat_name", "Unknown"),
                "productName": product_name,
                "productLink": product_url,
                "error_type": "product_page_error",
                "http_status": "N/A",
                "error_message": f"Error checking page: {str(e)}"
            }
            log_message(f"❌ Error checking product page {product_url}: {str(e)}", "ERROR")
            return None, error_info
        
        # Create dedicated crawler for this product
        try:
            async with AsyncWebCrawler(config=browser_config) as product_crawler:
                venue_session_id = f"{session_id}_{hash(product_url)}"
                log_message(f"📥 Processing product page for PDFs: {product_url}", "INFO")
                
                try:
                    # Add random delay to be polite to servers
                    await asyncio.sleep(random.uniform(5, 15))
                    
                    summary = await download_pdf_links(
                        product_crawler,
                        product_url=product_url,
                        output_folder="output",
                        pdf_selector=site["pdf_selector"],
                        session_id=venue_session_id,
                        regex_strategy=regex_strategy,
                        domain_name=domain_name,
                        pdf_llm_strategy=pdf_llm_strategy,
                        api_key=api_key,
                        cat_name=site["cat_name"],
                        pdf_button_selector=site.get("pdf_button_selector", ""),
                        preserve_product_name=preserve_product_name,
                    )
                    
                    if summary:
                        return {
                            "productLink": summary.get("productLink"),
                            "productName": summary.get("productName"),
                            "category": summary.get("category"),
                            "saved_count": summary.get("saved_count", 0),
                            "has_datasheet": summary.get("has_datasheet", False),
                        }, None
                    else:
                        # No PDFs found or processing failed
                        error_info = {
                            "page_number": page_number,
                            "cat_name": site.get("cat_name", "Unknown"),
                            "productName": product_name,
                            "productLink": product_url,
                            "error_type": "pdf_processing_failed",
                            "http_status": "N/A",
                            "error_message": "No PDFs found or processing returned no summary"
                        }
                        return None, error_info
                    
                except Exception as pdf_error:
                    error_info = {
                        "page_number": page_number,
                        "cat_name": site.get("cat_name", "Unknown"),
                        "productName": product_name,
                        "productLink": product_url,
                        "error_type": "pdf_download_error",
                        "http_status": "N/A",
                        "error_message": f"PDF processing exception: {str(pdf_error)}"
                    }
                    log_message(f"❌ PDF processing failed for {product_name}: {str(pdf_error)}", "ERROR")
                    return None, error_info
                    
        except Exception as crawler_error:
            error_info = {
                "page_number": page_number,
                "cat_name": site.get("cat_name", "Unknown"),
                "productName": product_name,
                "productLink": product_url,
                "error_type": "crawler_initialization_error",
                "http_status": "N/A",
                "error_message": f"Crawler error: {str(crawler_error)}"
            }
            log_message(f"❌ Crawler error for {product_name}: {str(crawler_error)}", "ERROR")
            return None, error_info

async def crawl_from_sites_csv(input_file: str, api_key: str = None, model: str = "groq/llama-3.1-8b-instant", 
                              status_callback=None, stop_requested_callback=None):
    """
    Crawl venue data from a list of category/product links, selectors, and button selectors stored in a CSV.
    
    Args:
        input_file (str): Path to the CSV file containing site configurations
        api_key (str): API key for the LLM provider. If None, will try to get from environment.
        model (str): LLM model to use. Defaults to "groq/deepseek-r1-distill-llama-70b".
        status_callback (function): Callback to update status for web interface
        stop_requested_callback (function): Callback to check if stop is requested
    """
    browser_config = get_browser_config()
    llm_strategy = get_llm_strategy(api_key=api_key, model=model)
    pdf_llm_strategy = get_pdf_llm_strategy(api_key=api_key, model=model)
    regex_strategy = get_regex_strategy()
    session_id = "bulk_crawl_session"

    all_venues = []
    # Track products per category for CSV export
    category_to_products = {}
    seen_links = set()
    
    # Statistics tracking
    stats = {
        "api_mode_count": 0,
        "html_mode_count": 0,
        "api_products": 0,
        "html_products": 0,
        "api_domains": [],
        "html_urls": []
    }
    
    # Concurrency control: limit parallel product processing
    # With 64GB RAM, we can handle 16-24 concurrent products safely
    MAX_CONCURRENT_PRODUCTS = int(os.environ.get('MAX_CONCURRENT_PRODUCTS', '20'))
    product_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PRODUCTS)

    sites = read_sites_from_csv(input_file)
    log_message(f"Loaded {len(sites)} sites to crawl.", "INFO")
    log_message(f"🔧 Configured for up to {MAX_CONCURRENT_PRODUCTS} concurrent product processing tasks", "INFO")

    # Update status for web interface
    if status_callback:
        status_callback({
            'total_sites': len(sites),
            'current_site': 0,
            'current_page': 1,
            'total_venues': 0
        })

    try:
        for index, site in enumerate(sites):
                cat_summaries = []
                # Check if stop is requested
                if stop_requested_callback and stop_requested_callback():
                    log_message("Stop requested by user.", "WARNING")
                    break

                mode = site.get("mode", "html")
                log_message(f"--- Crawling site {index+1}/{len(sites)} (Mode: {mode.upper()}) ---", "INFO")
                
                # Update status for web interface
                if status_callback:
                    status_callback({
                        'current_site': index + 1,
                        'current_page': 1
                    })

                # Dispatch based on mode
                if mode == "api":
                    # API-based product fetching using unified browser context
                    domain_name = site["domain_name"]
                    log_message(f"📡 API Mode: Fetching products for domain '{domain_name}'", "INFO")
                    stats["api_mode_count"] += 1
                    stats["api_domains"].append(domain_name)

                    # Create unified browser context for this domain
                    # This ensures all requests (API + product pages) use the same browser session
                    domain_url = f"https://{domain_name}" if not domain_name.startswith('http') else domain_name
                    log_message(f"🌐 Creating unified browser context for: {domain_url}", "INFO")
                    
                    browser_ctx = await create_unified_browser_context(domain_url)
                    
                    if not browser_ctx:
                        log_message(f"❌ Failed to create browser context for {domain_name}, skipping", "ERROR")
                        continue
                    
                    try:
                        cookies = browser_ctx.get('cookies', {})
                        headers = browser_ctx.get('headers', {})
                        log_message(f"✅ Unified browser context ready with {len(cookies)} cookies", "INFO")

                        start_page = site.get("start_page", 1)
                        end_page = site.get("end_page")
                        cur_page = start_page
                        total_api_products = 0
                        while True:
                            if end_page and cur_page > end_page:
                                break
                            
                            log_message(f"🟦 [API] Fetching page {cur_page} for domain '{domain_name}'", "INFO")
                            
                            try:
                                # Fetch products using the unified browser context
                                page_venues = await fetch_products_from_api_via_browser(
                                    domain_name=domain_name,
                                    page_number=cur_page,
                                    browser_context=browser_ctx
                                )
                                
                                if not page_venues:
                                    log_message(f"✅ No products or end of pagination for domain '{domain_name}' at page {cur_page}", "INFO")
                                    break
                                stats["api_products"] += len(page_venues)
                                total_api_products += len(page_venues)
                                log_message(f"✅ Fetched {len(page_venues)} products from API page {cur_page} for '{domain_name}'", "SUCCESS")
                                
                                # Error tracking for this page
                                page_errors = []
                                
                                # Process batch as before
                                BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '1'))
                                total_products = len(page_venues)
                                product_idx = 0
                                batch_num = 1
                                while product_idx < total_products:
                                    batch_venues = page_venues[product_idx : product_idx + BATCH_SIZE]
                                    log_message(f"⚡ [API] Processing batch {batch_num} (page {cur_page}, {len(batch_venues)} products)", "INFO")
                                    product_tasks = [
                                        process_single_product(
                                            venue=venue,
                                            site=site,
                                            browser_config=browser_config,
                                            pdf_llm_strategy=pdf_llm_strategy,
                                            regex_strategy=regex_strategy,
                                            api_key=api_key,
                                            session_id=session_id,
                                            domain_name=domain_name,
                                            semaphore=product_semaphore,
                                            stop_requested_callback=stop_requested_callback,
                                            page_number=cur_page,
                                            cookies=cookies,
                                            headers=headers,
                                            preserve_product_name=venue.get('productName'),  # Preserve API product name
                                        )
                                        for venue in batch_venues
                                    ]
                                    
                                    results = await asyncio.gather(*product_tasks, return_exceptions=True)
                                    for result in results:
                                        if isinstance(result, Exception):
                                            log_message(f"❌ Product processing raised exception: {str(result)}", "ERROR")
                                            # Create error entry for exception
                                            page_errors.append({
                                                "page_number": cur_page,
                                                "cat_name": site.get("cat_name", "Unknown"),
                                                "productName": "Unknown",
                                                "productLink": "Unknown",
                                                "error_type": "processing_exception",
                                                "http_status": "N/A",
                                                "error_message": f"Exception: {str(result)}"
                                            })
                                        elif result:
                                            # Result is tuple: (summary, error)
                                            summary, error = result
                                            if summary:
                                                cat_summaries.append(summary)
                                            if error:
                                                page_errors.append(error)
                                    log_message(f"✅ [API] Completed batch {batch_num} ({min(product_idx+BATCH_SIZE, total_products)}/{total_products} products) on page {cur_page}", "INFO")
                                    product_idx += BATCH_SIZE
                                    batch_num += 1
                                    if product_idx < total_products:
                                        await asyncio.sleep(5)
                                log_message(f"✅ [API] Completed all batches for {len(page_venues)} products on page {cur_page}", "INFO")
                                
                                # Write per-page error report if there are errors
                                if page_errors:
                                    try:
                                        os.makedirs("ERROR_REPORTS", exist_ok=True)
                                        from datetime import datetime
                                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        domain_sanitized = sanitize_folder_name(domain_name)
                                        error_csv = os.path.join("ERROR_REPORTS", f"errors_{domain_sanitized}_page_{cur_page}_{ts}.csv")
                                        
                                        with open(error_csv, "w", newline="", encoding="utf-8") as f:
                                            fieldnames = ["page_number", "cat_name", "productName", "productLink", "error_type", "http_status", "error_message"]
                                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                                            writer.writeheader()
                                            for error in page_errors:
                                                writer.writerow(error)
                                        
                                        log_message(f"📋 [API] Wrote error report: {error_csv} ({len(page_errors)} errors)", "WARNING")
                                    except Exception as e:
                                        log_message(f"⚠️ Failed to write error report for page {cur_page}: {e}", "ERROR")
                                else:
                                    log_message(f"✅ [API] No errors on page {cur_page}", "INFO")
                                
                                for v in page_venues:
                                    v["category"] = site["cat_name"]
                                    category_to_products.setdefault(site["cat_name"], []).append(v)
                                all_venues.extend(page_venues)
                                if status_callback:
                                    status_callback({
                                        'total_venues': len(all_venues),
                                        'current_page': cur_page
                                    })
                            except Exception as e:
                                log_message(f"❌ Unexpected error fetching page {cur_page} for {domain_name}: {str(e)}", "ERROR")
                                break
                            cur_page += 1
                        
                        log_message(f"✅ [API] Fetched and processed {total_api_products} total products for '{domain_name}'", "SUCCESS")
                    
                    finally:
                        # Always cleanup the browser context
                        log_message(f"🧹 Cleaning up unified browser context for {domain_name}", "INFO")
                        await close_unified_browser_context(browser_ctx)
                    
                    # Tag venues with category
                    for v in all_venues:
                        v["category"] = site["cat_name"]
                        category_to_products.setdefault(site["cat_name"], []).append(v)
                    all_venues.extend(all_venues) # This line seems redundant, but keeping as per original
                    
                    # Update total venues count
                    if status_callback:
                        status_callback({
                            'total_venues': len(all_venues)
                        })
                    
                else:
                    # HTML-based crawling (legacy mode)
                    url = site["url"]
                    css_selector = site["css_selector"]
                    button_selector = site["button_selector"]
                    
                    log_message(f"🌐 HTML Mode: Crawling URL '{url}'", "INFO")
                    
                    # Track HTML mode usage
                    stats["html_mode_count"] += 1
                    stats["html_urls"].append(url)

                    parsed = urlparse(url)
                    domain_name = parsed.netloc
                    log_message(f"Domain: {domain_name}", "INFO")
                    print(f"Domain: {domain_name}")

                    # Enhanced pagination handling
                    page_number = get_page_number(url)
                    
                    # Detect pagination type for better handling
                    pagination_type = detect_pagination_type(url)
                    log_message(f"🔍 Detected pagination type: {pagination_type}", "INFO")
                    
                    async with AsyncWebCrawler(config=browser_config) as crawler:
                        while True:
                            # Check if stop is requested
                            if stop_requested_callback and stop_requested_callback():
                                log_message("Stop requested by user.", "WARNING")
                                break

                            if button_selector:
                                # For button-based pagination, use the current URL
                                paged_url = url
                                log_message(f"🔄 Crawling URL with JS pagination: {paged_url} (page {page_number})", "INFO")
                                # Use JS-based extraction
                                venues, no_results = await fetch_and_process_page_with_js(
                                    crawler=crawler,
                                    page_url=paged_url,
                                    llm_strategy=llm_strategy,
                                    button_selector=button_selector,
                                    elements=css_selector,
                                    required_keys=REQUIRED_KEYS,
                                    seen_names=seen_links,
                                )
                            else:
                                # For URL-based pagination, construct the paged URL
                                paged_url = append_page_param(url, page_number, pagination_type)
                                log_message(f"🔄 Crawling URL with {pagination_type} pagination: {paged_url} (page {page_number})", "INFO")
                                # Use standard extraction
                                venues, no_results = await fetch_and_process_page(
                                    crawler = crawler,
                                    css_selector = css_selector,
                                    page_number = page_number,
                                    url = paged_url,
                                    llm_strategy = llm_strategy,
                                    session_id = f"{session_id}_{page_number}",
                                    required_keys = REQUIRED_KEYS,
                                    seen_names = seen_links,
                                )

                            if no_results or not venues:
                                log_message(f"🏁 Stopping pagination - no more results on page {page_number}", "INFO")
                                break
                            
                            # Process products in parallel with controlled concurrency
                            if stop_requested_callback and stop_requested_callback():
                                log_message("Stop requested by user.", "WARNING")
                                break
                            
                            log_message(f"🚀 Starting parallel processing of {len(venues)} products", "INFO")
                            
                            # Error tracking for HTML mode page
                            page_errors = []
                            
                            BATCH_SIZE = 1  # Configurable: how many products to launch per batch
                            total_products = len(venues)
                            product_idx = 0
                            batch_num = 1
                            while product_idx < total_products:
                                batch_venues = venues[product_idx : product_idx + BATCH_SIZE]
                                log_message(f"⚡ Processing batch {batch_num} ({len(batch_venues)} products)", "INFO")

                                product_tasks = [
                                    process_single_product(
                                        venue=venue,
                                        site=site,
                                        browser_config=browser_config,
                                        pdf_llm_strategy=pdf_llm_strategy,
                                        regex_strategy=regex_strategy,
                                        api_key=api_key,
                                        session_id=session_id,
                                        domain_name=domain_name,
                                        semaphore=product_semaphore,
                                        stop_requested_callback=stop_requested_callback,
                                        page_number=page_number
                                    )
                                    for venue in batch_venues
                                ]

                                results = await asyncio.gather(*product_tasks, return_exceptions=True)
                                for result in results:
                                    if isinstance(result, Exception):
                                        log_message(f"❌ Product processing raised exception: {str(result)}", "ERROR")
                                        page_errors.append({
                                            "page_number": page_number,
                                            "cat_name": site.get("cat_name", "Unknown"),
                                            "productName": "Unknown",
                                            "productLink": "Unknown",
                                            "error_type": "processing_exception",
                                            "http_status": "N/A",
                                            "error_message": f"Exception: {str(result)}"
                                        })
                                    elif result:
                                        summary, error = result
                                        if summary:
                                            cat_summaries.append(summary)
                                        if error:
                                            page_errors.append(error)

                                log_message(f"✅ Completed batch {batch_num} ({product_idx+BATCH_SIZE if product_idx+BATCH_SIZE < total_products else total_products}/{total_products} products)", "INFO")
                                product_idx += BATCH_SIZE
                                batch_num += 1
                                if product_idx < total_products:
                                    await asyncio.sleep(5)  # Brief pause between batches
                            
                            log_message(f"✅ Completed all batches for {len(venues)} products", "INFO")

                            # Write per-page error report for HTML mode if there are errors
                            if page_errors:
                                try:
                                    os.makedirs("ERROR_REPORTS", exist_ok=True)
                                    from datetime import datetime
                                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    domain_sanitized = sanitize_folder_name(domain_name)
                                    error_csv = os.path.join("ERROR_REPORTS", f"errors_{domain_sanitized}_page_{page_number}_{ts}.csv")
                                    
                                    with open(error_csv, "w", newline="", encoding="utf-8") as f:
                                        fieldnames = ["page_number", "cat_name", "productName", "productLink", "error_type", "http_status", "error_message"]
                                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                                        writer.writeheader()
                                        for error in page_errors:
                                            writer.writerow(error)
                                    
                                    log_message(f"📋 [HTML] Wrote error report: {error_csv} ({len(page_errors)} errors)", "WARNING")
                                except Exception as e:
                                    log_message(f"⚠️ Failed to write error report for page {page_number}: {e}", "ERROR")
                            else:
                                log_message(f"✅ [HTML] No errors on page {page_number}", "INFO")

                            # Track HTML products count
                            stats["html_products"] += len(venues)
                            
                            # Tag venues with category and collect for CSV export
                            for v in venues:
                                v["category"] = site["cat_name"]
                                category_to_products.setdefault(site["cat_name"], []).append(v)
                            all_venues.extend(venues)
                            
                            # Update total venues count for web interface
                            if status_callback:
                                status_callback({
                                    'total_venues': len(all_venues)
                                })


                            if page_number is not None:
                            # Increment page number for next iteration
                                page_number += 1
                            
                            # Update current page for web interface
                            if status_callback:
                                status_callback({
                                    'current_page': page_number
                                })
                            
                            await asyncio.sleep(random.uniform(3, 15))  # Be polite

                            if crawler is not None:
                                log_message(f"🔄 Restarting Crawler to prevent resource exhaustion", "INFO")
                                await crawler.close()
                                await asyncio.sleep(2)
                            crawler = AsyncWebCrawler(config=browser_config)
                            await crawler.start()
                            log_message("🚀 Fresh Crawler instance started", "INFO")

        # After finishing each site/category, cleanup and write per-category CSV
                try:
                    # Cleanup unfinished PDFs for this category
                    log_message("🧹 Cleaning up unfinished PDFs for category...", "INFO")
                    await _pdf_tracker.cleanup_unfinished()
                    
                    # Get and log tracker statistics
                    stats = await _pdf_tracker.get_stats()
                    log_message(f"📊 PDF Tracker Stats: {stats['total_pdfs']} total, {stats['cleaned']} cleaned, {stats['in_progress']} in progress, {stats['pending_copies']} pending copies", "INFO")
                    
                    if cat_summaries:
                        os.makedirs("CSVS", exist_ok=True)
                        from datetime import datetime
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        cat_name_sanitized = sanitize_folder_name(site["cat_name"]) if site.get("cat_name") else "Uncategorized"
                        out_csv = os.path.join("CSVS", f"downloaded_{cat_name_sanitized}_{ts}.csv")
                        with open(out_csv, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=["productLink","productName","category","saved_count","has_datasheet"])
                            writer.writeheader()
                            for row in cat_summaries:
                                writer.writerow(row)
                        log_message(f"🧾 Saved category summary: {out_csv}", "SUCCESS")
                except Exception as e:
                    log_message(f"⚠️ Failed to write category summary CSV: {e}", "WARNING")

    finally:
        # Final cleanup
        log_message("🧹 Final cleanup of PDF tracker...", "INFO")
        await _pdf_tracker.cleanup_unfinished()
        final_stats = await _pdf_tracker.get_stats()
        log_message(f"📊 Final PDF Tracker Stats: {final_stats}", "INFO")

    log_message(f"PDF LLM strategy usage: {pdf_llm_strategy.show_usage()}", "INFO")
    log_message(f"LLM strategy usage: {llm_strategy.show_usage()}", "INFO")
    log_message(f"Crawling completed. Total venues processed: {len(all_venues)}", "SUCCESS")
    
    # Log pipeline statistics
    log_message("=" * 60, "INFO")
    log_message("📊 PIPELINE STATISTICS SUMMARY", "INFO")
    log_message("=" * 60, "INFO")
    log_message(f"🔹 API Mode Sites: {stats['api_mode_count']}", "INFO")
    log_message(f"🔹 HTML Mode Sites: {stats['html_mode_count']}", "INFO")
    log_message(f"📦 API Products Fetched: {stats['api_products']}", "INFO")
    log_message(f"📦 HTML Products Crawled: {stats['html_products']}", "INFO")
    log_message(f"📊 Total Products: {stats['api_products'] + stats['html_products']}", "INFO")
    
    if stats['api_domains']:
        log_message(f"📡 API Domains Processed: {', '.join(stats['api_domains'])}", "INFO")
    if stats['html_urls']:
        log_message(f"🌐 HTML URLs Processed: {len(stats['html_urls'])} URL(s)", "INFO")
    
    log_message("=" * 60, "INFO")
async def main():
    log_message(f"{'='*50} Starting crawling {'='*50}", "INFO")
    # here we should take the csv file name from the .env file to enable the third crawler
    await crawl_from_sites_csv("sites.csv")
    log_message(f"{'='*50} Crawl completed {'='*50}", "INFO")

if __name__ == "__main__":
    asyncio.run(main()) 