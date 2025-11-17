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

            # PDF button selector for browser-triggered downloads
            pdf_button_selector = row.get('pdf_button_selector', '').strip()
            sites.append({
                "url": row["url"],
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
    stop_requested_callback=None
):
    """
    Process a single product: download and clean PDFs.
    Uses a semaphore to limit concurrency and avoid overwhelming servers.
    
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
        
    Returns:
        Summary dict or None if failed/stopped
    """
    async with semaphore:  # Control concurrency
        if stop_requested_callback and stop_requested_callback():
            log_message("Stop requested by user.", "WARNING")
            return None
        
        # Create dedicated crawler for this product
        async with AsyncWebCrawler(config=browser_config) as product_crawler:
            venue_session_id = f"{session_id}_{hash(venue['productLink'])}"
            log_message(f"📥 Processing product page for PDFs: {venue['productLink']}", "INFO")
            
            try:
                # Add random delay to be polite to servers
                await asyncio.sleep(random.uniform(5, 15))
                
                summary = await download_pdf_links(
                    product_crawler,
                    product_url=venue["productLink"],
                    output_folder="output",
                    pdf_selector=site["pdf_selector"],
                    session_id=venue_session_id,
                    regex_strategy=regex_strategy,
                    domain_name=domain_name,
                    pdf_llm_strategy=pdf_llm_strategy,
                    api_key=api_key,
                    cat_name=site["cat_name"],
                    pdf_button_selector=site.get("pdf_button_selector", ""),
                )
                
                if summary:
                    return {
                        "productLink": summary.get("productLink"),
                        "productName": summary.get("productName"),
                        "category": summary.get("category"),
                        "saved_count": summary.get("saved_count", 0),
                        "has_datasheet": summary.get("has_datasheet", False),
                    }
                return None
                
            except Exception as pdf_error:
                log_message(f"❌ PDF processing failed for {venue.get('productName', 'Unknown')}: {str(pdf_error)}", "ERROR")
                return None

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
                        
                        # Process products in parallel with controlled concurrency
                        if stop_requested_callback and stop_requested_callback():
                            log_message("Stop requested by user.", "WARNING")
                            break
                        
                        log_message(f"🚀 Starting parallel processing of {len(venues)} products", "INFO")
                        
                        BATCH_SIZE = 5  # Configurable: how many products to launch per batch
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
                                    stop_requested_callback=stop_requested_callback
                                )
                                for venue in batch_venues
                            ]

                            summaries = await asyncio.gather(*product_tasks, return_exceptions=True)
                            for summary in summaries:
                                if isinstance(summary, Exception):
                                    log_message(f"❌ Product processing raised exception: {str(summary)}", "ERROR")
                                elif summary:
                                    cat_summaries.append(summary)

                            log_message(f"✅ Completed batch {batch_num} ({product_idx+BATCH_SIZE if product_idx+BATCH_SIZE < total_products else total_products}/{total_products} products)", "INFO")
                            product_idx += BATCH_SIZE
                            batch_num += 1
                            if product_idx < total_products:
                                await asyncio.sleep(5)  # Brief pause between batches
                        
                        log_message(f"✅ Completed all batches for {len(venues)} products", "INFO")

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
async def main():
    log_message(f"{'='*50} Starting crawling {'='*50}", "INFO")
    # here we should take the csv file name from the .env file to enable the third crawler
    await crawl_from_sites_csv("sites.csv")
    log_message(f"{'='*50} Crawl completed {'='*50}", "INFO")

if __name__ == "__main__":
    asyncio.run(main()) 