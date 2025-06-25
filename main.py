import asyncio
import csv
import random

from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv

from utils.scraper_utils import download_pdf_links
from config import REQUIRED_KEYS
from utils.data_utils import save_venues_to_csv
from utils.scraper_utils import (
    fetch_and_process_page,
    get_browser_config,
    get_llm_strategy,
    get_regex_strategy,
    append_page_param,
    smart_pagination_crawl,
    PaginationHandler,
)


load_dotenv()


async def crawl_from_csv(input_file: str, output_file: str):
    """
    Crawl venue data from a list of category/product links stored in a CSV.
    Uses enhanced smart pagination system for better crawling across different websites.
    """
    browser_config = get_browser_config()
    llm_strategy = get_llm_strategy()
    regex_strategy = get_regex_strategy()
    session_id = "bulk_crawl_session"

    all_venues = []
    seen_names = set()

    # Read all links from the input CSV
    # in the future we will use datastructure like set to avoid duplicates
    # and make all the process faster and only outputs a folder of pdf's
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        links = [row[0] for row in reader if row]  # Skip empty rows
        print("------------------------------------------------------------------------------------------------------------------------------------")

    print(f"Loaded {len(links)} links to crawl.")

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for index, link in enumerate(links):
            print(f"\n{'='*80}")
            print(f"🚀 Crawling link {index+1}/{len(links)}")
            print(f"📍 URL: {link}")
            print(f"{'='*80}")

            # Use smart pagination crawler for each link
            try:
                venues = await smart_pagination_crawl(
                    crawler=crawler,
                    base_url=link,
                    llm_strategy=llm_strategy,
                    session_id=session_id,
                    required_keys=REQUIRED_KEYS,
                    seen_names=seen_names,
                    max_pages=50,  # Limit pages per category
                    pagination_handler=PaginationHandler()
                )
                
                print(f"\n📊 Processing {len(venues)} venues for PDF downloads...")
                
                # Download PDFs for each venue
                for venue_index, venue in enumerate(venues):
                    print(f"\n📄 Downloading PDFs for venue {venue_index+1}/{len(venues)}: {venue['productName']}")
                    
                    await download_pdf_links(
                        crawler, 
                        product_url=venue["productLink"],
                        product_name=venue["productName"],
                        output_folder="pdfs",
                        session_id="pdf_download_session",
                        regex_strategy=regex_strategy,
                    )
                    
                    # Add small delay between PDF downloads
                    await asyncio.sleep(random.uniform(1, 3))
                
                all_venues.extend(venues)
                print(f"✅ Completed crawling for link {index+1}. Total venues so far: {len(all_venues)}")
                
            except Exception as e:
                print(f"❌ Error crawling link {index+1}: {e}")
                continue
            
            # Add delay between different links
            if index < len(links) - 1:  # Don't delay after the last link
                delay = random.uniform(5, 10)
                print(f"⏳ Waiting {delay:.1f} seconds before next link...")
                await asyncio.sleep(delay)
            
    # Save all collected venues
    save_venues_to_csv(all_venues, output_file)
    print(f"\n🎉 Crawling completed! Saved {len(all_venues)} products to '{output_file}'.")
    llm_strategy.show_usage()


async def main():
    await crawl_from_csv("D:/projects/deepseek-ai-web-crawler/stMotor.tsv", "output.csv")


if __name__ == "__main__":
    asyncio.run(main())
