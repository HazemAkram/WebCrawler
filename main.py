import asyncio
import csv
import random

from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv

from urllib.parse import parse_qs, urlparse

from utils.scraper_utils import download_pdf_links
from config import REQUIRED_KEYS
from utils.scraper_utils import (
    fetch_and_process_page,
    get_browser_config,
    get_llm_strategy,
    append_page_param,  
    get_regex_strategy,
    fetch_and_process_page_with_js,  # NEW: import the JS-based extraction
    get_page_number
)

load_dotenv()

def read_sites_from_csv(input_file):
    sites = []
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            sites.append({
                "url": row["url"],
                "css_selector": [s.strip() for s in row['css_selector'].split(',') if s.strip()],
                "button_selector": row["button_selector"],
            })
    return sites

async def crawl_from_sites_csv(input_file: str):
    """
    Crawl venue data from a list of category/product links, selectors, and button selectors stored in a CSV.
    """
    browser_config = get_browser_config()
    llm_strategy = get_llm_strategy()
    regex_strategy = get_regex_strategy()
    session_id = "bulk_crawl_session"

    all_venues = []
    seen_names = set()

    sites = read_sites_from_csv(input_file)
    print(f"Loaded {len(sites)} sites to crawl.")

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for index, site in enumerate(sites):
            url = site["url"]
            css_selector = site["css_selector"]
            button_selector = site["button_selector"]
            print(f"\n--- Crawling site {index+1}/{len(sites)} ---")

            parsed = urlparse(url)
            domain_name = parsed.netloc
            print(f"domain : {domain_name}")

            page_number = get_page_number(url)
            while True:

                if button_selector:
                    paged_url = url
                    print(f"🔄 Crawling URL: {paged_url}")
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

                    paged_url = append_page_param(url, page_number)
                    print(f"🔄 Crawling URL: {paged_url}")
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
                    print(f"🏁 Stopping pagination - no more results on page {page_number}")
                    break

                for venue in venues:
                    await asyncio.sleep(random.uniform(5, 15))
                    venue_session_id = f"{session_id}_{venue['productName']}"
                    await download_pdf_links(
                        crawler, 
                        product_url=venue["productLink"],
                        product_name=venue["productName"],
                        output_folder="output",
                        session_id=venue_session_id,
                        regex_strategy=regex_strategy,
                        domain_name=domain_name
                    )

                all_venues.extend(venues)
                if page_number == None: 
                    break
                page_number += 1
                await asyncio.sleep(random.uniform(3, 15))  # Be polite

    llm_strategy.show_usage()

async def main():
    await crawl_from_sites_csv("sites.csv")

if __name__ == "__main__":
    asyncio.run(main())
