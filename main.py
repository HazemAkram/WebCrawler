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

from utils.scraper_utils import download_pdf_links, set_log_callback as set_scraper_log_callback
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
    detect_pagination_type
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
            sites.append({
                "url": row["url"],
                "css_selector": [s.strip() for s in row['css_selector'].split('|') if s.strip()],
                "pdf_selector": [s.strip() for s in row['pdf_selector'].split('|') if s.strip()],  # Add pdf_selector with fallback
                "button_selector": row["button_selector"],
                
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
    seen_names = set()
    
    # PDF crawler management
    pdf_crawler = None
    products_processed_with_pdf_crawler = 0
    MAX_PRODUCTS_PER_PDF_CRAWLER = 10  # Restart PDF crawler every 10 products

    sites = read_sites_from_csv(input_file)
    log_message(f"Loaded {len(sites)} sites to crawl.", "INFO")
    log_message(f"Site: {sites}", "INFO")

    # Update status for web interface
    if status_callback:
        status_callback({
            'total_sites': len(sites),
            'current_site': 0,
            'current_page': 1,
            'total_venues': 0
        })

    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            for index, site in enumerate(sites):
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
                if page_number is None:
                    page_number = 1  # Start from page 1 if no page number found
                
                # Detect pagination type for better handling
                pagination_type = detect_pagination_type(url)
                log_message(f"🔍 Detected pagination type: {pagination_type}", "INFO")
                
                count = 0
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
                            seen_names=seen_names,
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
                            seen_names = seen_names,
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

                        await asyncio.sleep(random.uniform(5, 15))
                        venue_session_id = f"{session_id}_{venue['productName']}"
                        
                        log_message(f"📥 Downloading PDFs for: {venue['productName']}", "INFO")
                        
                        try:
                            await download_pdf_links(
                                pdf_crawler,  # Use dedicated PDF crawler
                                product_url=venue["productLink"],
                                product_name=venue["productName"],
                                output_folder="output",
                                pdf_selector=site["pdf_selector"],  # Add pdf_selector from CSV
                                session_id=venue_session_id,
                                regex_strategy=regex_strategy,
                                domain_name=domain_name,
                                pdf_llm_strategy=pdf_llm_strategy,
                                api_key=api_key,
                            )
                            products_processed_with_pdf_crawler += 1
                            
                        except Exception as pdf_error:
                            log_message(f"❌ PDF processing failed for {venue['productName']}: {str(pdf_error)}", "ERROR")
                            # Don't increment counter on failure, but continue processing

                        await cleanup_crawler_session(pdf_crawler, venue_session_id)

                        count += 1
                        
                        if count % 10 == 0:
                            # Force to Collect garbage after every 10 products.
                            gc.collect()
                            log_message(f"🧹 Collected garbage after {count} products", "INFO")

                    all_venues.extend(venues)
                    
                    # Update total venues count for web interface
                    if status_callback:
                        status_callback({
                            'total_venues': len(all_venues)
                        })

                    # Increment page number for next iteration
                    page_number += 1
                    
                    # Update current page for web interface
                    if status_callback:
                        status_callback({
                            'current_page': page_number
                        })
                    
                    await asyncio.sleep(random.uniform(3, 15))  # Be polite

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