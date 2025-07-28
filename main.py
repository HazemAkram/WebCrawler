import asyncio
import csv
import random

from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv

from utils.scraper_utils import download_pdf_links
from config import REQUIRED_KEYS
from utils.scraper_utils import (
    fetch_and_process_page,
    get_browser_config,
    get_llm_strategy,
    append_page_param,  
    get_regex_strategy,
)


load_dotenv()


async def crawl_from_csv(input_file: str):
    """
    Crawl venue data from a list of category/product links stored in a CSV.
    """
    browser_config = get_browser_config()
    llm_strategy = get_llm_strategy()
    regex_strategy = get_regex_strategy()
    session_id = "bulk_crawl_session"
    css_selector = ["td.product-name"]  # CSS selector to target specific content on the page

    all_venues = []
    seen_names = set()

    # Read all links from the input CSV
    # in the futer we will use datastructure like set to avoid duplicates
    # and make all the process faster and only outputs a folder of pdf's
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        links = [row[0] for row in reader if row]  # Skip empty rows
        print("------------------------------------------------------------------------------------------------------------------------------------")

    print(f"Loaded {len(links)} links to crawl.")

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for index, link in enumerate(links):
            print(f"\n--- Crawling link {index+1}/{len(links)} ---\n{link}")
            
            page_number = 1

            while True:
                paged_url = append_page_param(link, page_number)
                print(f"🔄 Crawling URL: {paged_url}")
                

                venues, no_results = await fetch_and_process_page(
                    crawler = crawler,
                    css_selector = css_selector,
                    page_number = page_number ,
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
                    session_id = f"{session_id}_{venue['productName']}"
                    await download_pdf_links(
                        crawler, 
                        product_url=venue["productLink"],
                        product_name=venue["productName"],
                        output_folder="pdfs",
                        session_id=session_id,
                        regex_strategy=regex_strategy,
                    )

                all_venues.extend(venues)
                page_number += 1
                await asyncio.sleep(random.uniform(3, 15))  # Be polite
            
    # Save all collected venues
    llm_strategy.show_usage()


async def main():
    await crawl_from_csv("D:/projects/deepseek-ai-web-crawler/stMotor.tsv")


if __name__ == "__main__":
    asyncio.run(main())
