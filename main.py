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
from collections import defaultdict
from typing import Dict, Set

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv
from urllib.parse import urlparse

from utils.scraper_utils import download_pdf_links, set_log_callback as set_scraper_log_callback, get_host, add_jitter_delay
from config import REQUIRED_KEYS, DEFAULT_CONFIG
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
    sanitize_folder_name
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


class SharedResources:
    """
    Shared resources for parallel crawling operations.
    Manages browser pools, sessions, and semaphores.
    """
    def __init__(self, api_key: str, model: str, config: dict):
        self.api_key = api_key
        self.model = model
        self.config = config
        self.concurrency_config = config.get("concurrency", {})
        
        # Semaphores for rate limiting
        self.site_semaphore = asyncio.Semaphore(self.concurrency_config.get("max_concurrent_sites", 4))
        self.product_semaphore = asyncio.Semaphore(self.concurrency_config.get("max_products_per_site", 8))
        self.download_semaphore = asyncio.Semaphore(self.concurrency_config.get("max_concurrent_downloads", 16))
        
        # Per-domain semaphores for rate limiting
        self.domain_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.per_domain_limit = self.concurrency_config.get("per_domain_limit", 2)
        
        # Shared HTTP session
        self.http_session: aiohttp.ClientSession = None
        
        # Browser pool
        self.browser_pool: list = []
        self.browser_config = get_browser_config()
        self.max_browser_pool = self.concurrency_config.get("max_browser_pool_per_worker", 2)
        
        # Strategies (shared across all operations)
        self.llm_strategy = get_llm_strategy(api_key=api_key, model=model)
        self.pdf_llm_strategy = get_pdf_llm_strategy(api_key=api_key, model=model)
        self.regex_strategy = get_regex_strategy()
        
        # Tracking
        self.seen_links: Set[str] = set()
        self.all_venues = []
        self.category_to_products = {}
        
    def get_domain_semaphore(self, url: str) -> asyncio.Semaphore:
        """Get or create a semaphore for a specific domain."""
        host = get_host(url)
        if host not in self.domain_semaphores:
            self.domain_semaphores[host] = asyncio.Semaphore(self.per_domain_limit)
        return self.domain_semaphores[host]
    
    async def initialize(self):
        """Initialize shared resources."""
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=90, connect=15, sock_read=60)
        self.http_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=False, limit=100),
            timeout=timeout
        )
        log_message("✅ Shared HTTP session initialized", "INFO")
    
    async def cleanup(self):
        """Cleanup shared resources."""
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
            log_message("✅ Shared HTTP session closed", "INFO")
        
        # Close any remaining browsers in pool
        for browser in self.browser_pool:
            try:
                await browser.close()
            except Exception:
                pass
        self.browser_pool.clear()
        log_message("✅ Browser pool cleaned up", "INFO")
    
    async def get_browser(self) -> AsyncWebCrawler:
        """Get a browser from the pool or create a new one."""
        # Try to reuse from pool
        if self.browser_pool:
            return self.browser_pool.pop(0)
        
        # Create new browser
        browser = AsyncWebCrawler(config=self.browser_config)
        await browser.start()
        return browser
    
    async def return_browser(self, browser: AsyncWebCrawler):
        """Return a browser to the pool."""
        if len(self.browser_pool) < self.max_browser_pool:
            self.browser_pool.append(browser)
        else:
            # Pool is full, close this browser
            try:
                await browser.close()
            except Exception:
                pass

def read_sites_from_csv(input_file):
    log_message(f"🔄 Reading sites from CSV: {input_file}", "INFO")
    sites = []
    with open(input_file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            css_list = [s.strip() for s in row['css_selector'].split('|') if s.strip()] if row.get('css_selector') else []
            pdf_list = [s.strip() for s in row['pdf_selector'].split('|') if s.strip()] if row.get('pdf_selector') else []
            # Optional name selector to extract product name on product page
            name_selector = row.get('name_selector', '').strip()
            if name_selector:
                # Ensure name selector is available to the PDF LLM by adding it to the target elements list
                pdf_list.append(name_selector)

            sites.append({
                "url": row["url"],
                "cat_name": row.get("cat_name", "Uncategorized"),
                "css_selector": css_list,
                "pdf_selector": pdf_list,
                "button_selector": row.get("button_selector", ""),
            })
    return sites

async def cleanup_crawler_session(crawler, session_id):
    """
    Clean up crawler session to prevent resource leaks.
    
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
    product_link: str,
    site: dict,
    shared_resources: SharedResources,
    stop_requested_callback=None
) -> dict:
    """
    Process a single product: download PDFs and clean them.
    
    Args:
        product_link: URL of the product page
        site: Site configuration dictionary
        shared_resources: Shared resources instance
        stop_requested_callback: Callback to check if stop is requested
        
    Returns:
        dict: Summary of processed product
    """
    if stop_requested_callback and stop_requested_callback():
        return None
    
    # Acquire product semaphore to limit concurrent products
    async with shared_resources.product_semaphore:
        # Get domain semaphore for rate limiting
        domain_semaphore = shared_resources.get_domain_semaphore(product_link)
        
        # Get a browser from the pool
        browser = await shared_resources.get_browser()
        
        try:
            # Add jitter delay for politeness
            await add_jitter_delay()
            
            parsed = urlparse(product_link)
            domain_name = parsed.netloc
            session_id = f"product_{hash(product_link)}"
            
            log_message(f"📥 Processing product: {product_link}", "INFO")
            
            summary = await download_pdf_links(
                browser,
                product_url=product_link,
                output_folder="output",
                pdf_selector=site["pdf_selector"],
                session_id=session_id,
                regex_strategy=shared_resources.regex_strategy,
                domain_name=domain_name,
                pdf_llm_strategy=shared_resources.pdf_llm_strategy,
                api_key=shared_resources.api_key,
                cat_name=site["cat_name"],
                client_session=shared_resources.http_session,
                domain_semaphore=domain_semaphore,
            )
            
            return summary
            
        except Exception as e:
            log_message(f"❌ Error processing product {product_link}: {str(e)}", "ERROR")
            return None
        finally:
            # Return browser to pool
            await shared_resources.return_browser(browser)


async def process_products_batch(
    product_links: list,
    site: dict,
    shared_resources: SharedResources,
    stop_requested_callback=None
) -> list:
    """
    Process a batch of products concurrently.
    
    Args:
        product_links: List of product URLs
        site: Site configuration dictionary
        shared_resources: Shared resources instance
        stop_requested_callback: Callback to check if stop is requested
        
    Returns:
        list: List of product summaries
    """
    tasks = []
    for product_link in product_links:
        if stop_requested_callback and stop_requested_callback():
            break
        task = asyncio.create_task(
            process_single_product(product_link, site, shared_resources, stop_requested_callback)
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None results and exceptions
    summaries = []
    for result in results:
        if isinstance(result, Exception):
            log_message(f"⚠️ Product processing exception: {str(result)}", "WARNING")
        elif result is not None:
            summaries.append(result)
    
    return summaries


async def process_single_site(
    site: dict,
    site_index: int,
    total_sites: int,
    shared_resources: SharedResources,
    status_callback=None,
    stop_requested_callback=None
) -> list:
    """
    Process a single site with all its pages and products.
    
    Args:
        site: Site configuration dictionary
        site_index: Index of the current site
        total_sites: Total number of sites to process
        shared_resources: Shared resources instance
        status_callback: Callback to update status
        stop_requested_callback: Callback to check if stop is requested
        
    Returns:
        list: List of product summaries for this site
    """
    if stop_requested_callback and stop_requested_callback():
        return []
    
    # Acquire site semaphore to limit concurrent sites
    async with shared_resources.site_semaphore:
        url = site["url"]
        css_selector = site["css_selector"]
        button_selector = site["button_selector"]
        cat_name = site["cat_name"]
        
        log_message(f"--- Crawling site {site_index+1}/{total_sites}: {cat_name} ---", "INFO")
        
        if status_callback:
            status_callback({
                'current_site': site_index + 1,
                'current_page': 1
            })
        
        parsed = urlparse(url)
        domain_name = parsed.netloc
        
        # Detect pagination
        page_number = get_page_number(url)
        pagination_type = detect_pagination_type(url)
        log_message(f"🔍 Detected pagination type: {pagination_type}", "INFO")
        
        # Get a browser for category page crawling
        browser = await shared_resources.get_browser()
        
        try:
            all_product_links = []
            session_id = f"site_{hash(url)}"
            
            # Create a local seen_links for this site to avoid race conditions with parallel sites
            local_seen_links = set()
            
            # Crawl all pages to collect product links
            while True:
                if stop_requested_callback and stop_requested_callback():
                    break
                
                # Add jitter delay for politeness
                await add_jitter_delay()
                
                if button_selector:
                    paged_url = url
                    log_message(f"🔄 Crawling URL with JS pagination: {paged_url}", "INFO")
                    venues, no_results = await fetch_and_process_page_with_js(
                        crawler=browser,
                        page_url=paged_url,
                        llm_strategy=shared_resources.llm_strategy,
                        button_selector=button_selector,
                        elements=css_selector,
                        required_keys=REQUIRED_KEYS,
                        seen_names=local_seen_links,
                    )
                else:
                    paged_url = append_page_param(url, page_number, pagination_type)
                    log_message(f"🔄 Crawling page {page_number}: {paged_url}", "INFO")
                    venues, no_results = await fetch_and_process_page(
                        crawler=browser,
                        css_selector=css_selector,
                        page_number=page_number,
                        url=paged_url,
                        llm_strategy=shared_resources.llm_strategy,
                        session_id=f"{session_id}_{page_number}",
                        required_keys=REQUIRED_KEYS,
                        seen_names=local_seen_links,
                    )
                
                if no_results or not venues:
                    log_message(f"🏁 No more results on page {page_number}", "INFO")
                    break
                
                # Collect product links
                for venue in venues:
                    product_link = venue.get("productLink")
                    if product_link and product_link not in local_seen_links:
                        all_product_links.append(product_link)
                        local_seen_links.add(product_link)
                        # Also add to global shared set for cross-site deduplication
                        shared_resources.seen_links.add(product_link)
                
                log_message(f"📊 Found {len(venues)} products on page {page_number}", "INFO")
                
                if page_number is not None:
                    page_number += 1
                else:
                    break  # No pagination, only one page
                
                if status_callback:
                    status_callback({'current_page': page_number})
            
            log_message(f"✅ Total products found for {cat_name}: {len(all_product_links)}", "INFO")
            
            # Process products in batches concurrently
            summaries = await process_products_batch(
                all_product_links,
                site,
                shared_resources,
                stop_requested_callback
            )
            
            # Write category summary CSV
            if summaries:
                try:
                    os.makedirs("CSVS", exist_ok=True)
                    from datetime import datetime
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cat_name_sanitized = sanitize_folder_name(cat_name)
                    out_csv = os.path.join("CSVS", f"downloaded_{cat_name_sanitized}_{ts}.csv")
                    with open(out_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["productLink", "productName", "category", "saved_count", "has_datasheet"])
                        writer.writeheader()
                        for row in summaries:
                            writer.writerow(row)
                    log_message(f"🧾 Saved category summary: {out_csv}", "SUCCESS")
                except Exception as e:
                    log_message(f"⚠️ Failed to write category summary CSV: {e}", "WARNING")
            
            return summaries
            
        finally:
            # Return browser to pool
            await shared_resources.return_browser(browser)
            gc.collect()


async def crawl_from_sites_csv_parallel(input_file: str, api_key: str = None, model: str = "groq/llama-3.1-8b-instant", 
                                       status_callback=None, stop_requested_callback=None):
    """
    Parallel version: Crawl sites with concurrent processing at site and product levels.
    
    Args:
        input_file (str): Path to the CSV file containing site configurations
        api_key (str): API key for the LLM provider
        model (str): LLM model to use
        status_callback (function): Callback to update status for web interface
        stop_requested_callback (function): Callback to check if stop is requested
    """
    sites = read_sites_from_csv(input_file)
    log_message(f"📦 Loaded {len(sites)} sites to crawl with parallel processing", "INFO")
    
    if status_callback:
        status_callback({
            'total_sites': len(sites),
            'current_site': 0,
            'current_page': 1,
            'total_venues': 0
        })
    
    # Initialize shared resources
    shared_resources = SharedResources(api_key, model, DEFAULT_CONFIG)
    await shared_resources.initialize()
    
    try:
        # Process sites concurrently with limited concurrency
        tasks = []
        for index, site in enumerate(sites):
            if stop_requested_callback and stop_requested_callback():
                break
            task = asyncio.create_task(
                process_single_site(
                    site, index, len(sites), shared_resources,
                    status_callback, stop_requested_callback
                )
            )
            tasks.append(task)
        
        # Wait for all site tasks to complete
        all_summaries = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        total_products = 0
        for summaries in all_summaries:
            if isinstance(summaries, Exception):
                log_message(f"⚠️ Site processing exception: {str(summaries)}", "WARNING")
            elif summaries:
                total_products += len(summaries)
        
        log_message(f"✅ Crawling completed. Total products processed: {total_products}", "SUCCESS")
        log_message(f"📊 LLM strategy usage: {shared_resources.llm_strategy.show_usage()}", "INFO")
        log_message(f"📊 PDF LLM strategy usage: {shared_resources.pdf_llm_strategy.show_usage()}", "INFO")
        
    finally:
        # Cleanup shared resources
        await shared_resources.cleanup()


async def crawl_from_sites_csv(input_file: str, api_key: str = None, model: str = "groq/llama-3.1-8b-instant", 
                              status_callback=None, stop_requested_callback=None):
    """
    Main entry point - delegates to parallel crawler.
    
    Args:
        input_file (str): Path to the CSV file containing site configurations
        api_key (str): API key for the LLM provider
        model (str): LLM model to use
        status_callback (function): Callback to update status for web interface
        stop_requested_callback (function): Callback to check if stop is requested
    """
    # Use the new parallel implementation
    await crawl_from_sites_csv_parallel(
        input_file=input_file,
        api_key=api_key,
        model=model,
        status_callback=status_callback,
        stop_requested_callback=stop_requested_callback
    )


async def crawl_from_sites_csv_sequential_legacy(input_file: str, api_key: str = None, model: str = "groq/llama-3.1-8b-instant", 
                              status_callback=None, stop_requested_callback=None):
    """
    LEGACY: Sequential crawler (kept for reference, not used).
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
    
    # PDF crawler management
    pdf_crawler = None
    products_processed_with_pdf_crawler = 0
    MAX_PRODUCTS_PER_PDF_CRAWLER = 10  # Restart PDF crawler every 10 products

    sites = read_sites_from_csv(input_file)
    log_message(f"Loaded {len(sites)} sites to crawl.", "INFO")

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

                url = site["url"]
                css_selector = site["css_selector"]
                button_selector = site["button_selector"]
                
                log_message(f"--- Crawling site {index+1}/{len(sites)} ---", "INFO")
                
                # Update status for web interface
                if status_callback:
                    status_callback({
                        'current_site': index + 1,
                        'current_page': 1
                    })

                parsed = urlparse(url)
                domain_name = parsed.netloc
                log_message(f"Domain: {domain_name}", "INFO")
                print(f"Domain: {domain_name}")

                # Enhanced pagination handling
                page_number = get_page_number(url)
                # if page_number is None:
                #     page_number = 9999  # Start from page 1 if no page number found
                
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
                        
                        
                        for venue in venues:
                            # Check if stop is requested
                            if stop_requested_callback and stop_requested_callback():
                                log_message("Stop requested by user.", "WARNING")
                                break

                            # Initialize or restart PDF crawler if needed
                            if pdf_crawler is None or products_processed_with_pdf_crawler >= MAX_PRODUCTS_PER_PDF_CRAWLER:
                                if pdf_crawler is not None:
                                    log_message("🔄 Restarting PDF crawler to prevent resource exhaustion", "INFO")
                                    await pdf_crawler.close()
                                    await asyncio.sleep(2)  # Brief pause to ensure cleanup
                                
                                pdf_crawler = AsyncWebCrawler(config=browser_config)
                                await pdf_crawler.start()
                                products_processed_with_pdf_crawler = 0
                                log_message("🚀 Fresh PDF crawler instance started", "INFO")

                            await asyncio.sleep(random.uniform(10, 25))
                            venue_session_id = f"{session_id}_{hash(venue['productLink'])}"
                            log_message(f"📥 Processing product page for PDFs: {venue['productLink']}", "INFO")
                            
                            try:
                                summary = await download_pdf_links(
                                    pdf_crawler,  # Use dedicated PDF crawler
                                    product_url=venue["productLink"],
                                    output_folder="output",
                                    pdf_selector=site["pdf_selector"],  # Add pdf_selector from CSV
                                    session_id=venue_session_id,
                                    regex_strategy=regex_strategy,
                                    domain_name=domain_name,
                                    pdf_llm_strategy=pdf_llm_strategy,
                                    api_key=api_key,
                                    cat_name=site["cat_name"],  # Add category name for folder organization
                                )
                                if summary:
                                    cat_summaries.append({
                                        "productLink": summary.get("productLink"),
                                        "productName": summary.get("productName"),
                                        "category": summary.get("category"),
                                        "saved_count": summary.get("saved_count", 0),
                                        "has_datasheet": summary.get("has_datasheet", False),
                                    })
                                products_processed_with_pdf_crawler += 1
                                
                            except Exception as pdf_error:
                                log_message(f"❌ PDF processing failed for {venue['productName']}: {str(pdf_error)}", "ERROR")
                                # Don't increment counter on failure, but continue processing

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

        # After finishing each site/category, write per-category CSV
                try:
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
        # Ensure PDF crawler is properly closed
        if pdf_crawler is not None:
            await pdf_crawler.close()
            log_message("🚪 PDF crawler properly closed", "INFO")

    log_message(f"PDF LLM strategy usage: {pdf_llm_strategy.show_usage()}", "INFO")
    log_message(f"LLM strategy usage: {llm_strategy.show_usage()}", "INFO")
    log_message(f"Crawling completed. Total venues processed: {len(all_venues)}", "SUCCESS")
async def main():
    log_message(f"{'='*50} Starting crawling {'='*50}", "INFO")
    await crawl_from_sites_csv("sites.csv")
    log_message(f"{'='*50} Crawl completed {'='*50}", "INFO")

if __name__ == "__main__":
    asyncio.run(main()) 