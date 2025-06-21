import json
import os

import aiofiles
import aiohttp
from urllib.parse import urljoin, urlparse

from typing import List, Set, Tuple
from fake_useragent import UserAgent
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    LLMConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMExtractionStrategy,
    RegexExtractionStrategy,
)


from models.venue import Venue
from utils.data_utils import is_complete_venue, is_duplicate_venue


# https://www.ors.com.tr/en/tek-sirali-sabit-bilyali-rulmanlar
async def download_pdf_links(
        crawler: AsyncWebCrawler, 
        product_url: str, 
        output_folder="downloads", 
        session_id="pdf_download_session", 
        regex_strategy: RegexExtractionStrategy = None 
        ):
    
    """
    Opens the given product page, searches for any PDF links, and downloads them.
    """

    print((f"\n---------------------------------------------------------------Entering product -------------------------------------------------------------------\n"))

    try:

        response = await crawler.arun(
        url=product_url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,  # Do not use cached data
            extraction_strategy=regex_strategy,  # Strategy for data extraction
            session_id=session_id,  # Unique session ID for the crawl
            css_selector= "a",  # Target specific content on the page (we don't use it now but maybe will use it as a tag selector in the future)
        ),
    )
        
        extracted_data = response.html
        # print(extracted_data)

        html = extracted_data
        print(type(html))
        soup = BeautifulSoup(html, "html.parser")
        print(type(soup))

        # page_content = response.content

        # Extract all <a> href links
        pdf_links = []

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                full_url = urljoin(product_url, href)
                pdf_links.append(full_url)
                print((f"\n---------------------------------------------------------------Found PDF link -------------------------------------------------------------------\n"))

        # Create the download folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Download each PDF
        async with aiohttp.ClientSession() as session:
            for pdf_url in pdf_links:
                filename = os.path.basename(urlparse(pdf_url).path)
                save_path = os.path.join(output_folder, filename)

                try:
                    async with session.get(pdf_url) as resp:
                        if resp.status == 200:
                            async with aiofiles.open(save_path, "wb") as f:
                                await f.write(await resp.read())
                            print(f"📄 Downloaded PDF: {save_path}")
                        else:
                            print(f"❌ Failed to download: {pdf_url}")
                except Exception as e:
                    print(f"❌ Error downloading {pdf_url}: {e}")

    except Exception as e:
        print(f"⚠️ Error processing {product_url}: {e}")

def append_page_param(base_url: str, page_number: int) -> str:
    """
    Appends a correct page param to any URL.
    """
    delimiter = '&' if '?' in base_url else '?'
    return f"{base_url}{delimiter}page={page_number}"


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
        headless=False,  # Whether to run in headless mode (no GUI)
        verbose=True,  # Enable verbose logging
        headers={"User-Agent": user_agent},  # Custom headers to include
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



def get_llm_strategy() -> LLMExtractionStrategy:
    """
    Returns the configuration for the language model extraction strategy.

    Returns:
        LLMExtractionStrategy: The settings for how to extract data using LLM.
    """
    # https://docs.crawl4ai.com/api/strategies/#llmextractionstrategy
    return LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider = "groq/deepseek-r1-distill-llama-70b", # LLM model to use
            api_token= "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",  # API token for the LLM provider        
                             ),
        schema=Venue.model_json_schema(),  # JSON schema of the data model
        extraction_type="schema",  # Type of extraction to perform
        instruction=(
            
            "You are tasked with extracting structured product information from this webpage. "
            "Your goal is to identify all products displayed in the main content or product grid. "
            "For each product, extract only the following fields:\n\n"
            "- productName: the full name or title of the product as shown to the user.\n"
            "- productLink: the full clickable URL or anchor link pointing to the product's detail page.\n\n"
            "Ignore navigation menus, footers, or sidebars. Focus only on real product listings. "
            "Do not guess missing data. Only include items that contain both the name and link.\n\n"
            "Output a list of dictionaries following the required schema."

        ),  # Instructions for the LLM
        input_format="markdown",  # Format of the input content
        verbose=True,  # Enable verbose logging
    )


async def check_no_results(
    crawler: AsyncWebCrawler,
    url: str,
    session_id: str,
) -> bool:
    """
    Checks if the "No Results Found" message is present on the page.

    Args:
        crawler (AsyncWebCrawler): The web crawler instance.
        url (str): The URL to check.
        session_id (str): The session identifier.

    Returns:
        bool: True if "No Results Found" message is found, False otherwise.
    """
    # Fetch the page without any CSS selector or extraction strategy
    result = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            session_id=session_id,
        ),
    )

    if result.success:
        if "No Results Found" in result.cleaned_html:
            return True
    else:
        print(
            f"Error fetching page for 'No Results Found' check: {result.error_message}"
        )

    return False


async def fetch_and_process_page(
    crawler: AsyncWebCrawler,
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
    print(f"Fetching page {page_number} from URL: {url}")    


    # Check if "No Results Found" message is present
    no_results = await check_no_results(crawler, url, session_id)

    if no_results:
        return [], True  # No more results, signal to stop crawling

    # Fetch page content with the extraction strategy
    result = await crawler.arun(
        url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,  # Do not use cached data
            extraction_strategy=llm_strategy,  # Strategy for data extraction
            # css_selector=css_selector,  # Target specific content on the page (we don't use it now but maybe will use it as a tag selector in the future)
            session_id=session_id,  # Unique session ID for the crawl
        ),
    )

    if not (result.success and result.extracted_content):
        print(f"Error fetching page {page_number}: {result.error_message}")
        return [], False

    # Parse extracted content
    extracted_data = json.loads(result.extracted_content)
    if not extracted_data:
        print(f"No products found on page {page_number}.")
        return [], False

    # After parsing extracted content
    print("Extracted data:", extracted_data)

    # Process product
    complete_venues = []
    for venue in extracted_data:
        # Debugging: Print each venue to understand its structure
        print("Processing venue:", venue)

        # Ignore the 'error' key if it's False
        if venue.get("error") is False:
            venue.pop("error", None)  # Remove the 'error' key if it's False

        if not is_complete_venue(venue, required_keys):
            continue  # Skip incomplete venues

        if is_duplicate_venue(venue["productName"], seen_names):
            print(f"Duplicate venue '{venue['productName']}' found. Skipping.")
            continue  # Skip duplicate venues

        # Add venue to the list
        seen_names.add(venue["productName"])

        if "productLink" in venue:
            venue["productLink"] = venue["productLink"].replace("/en/en/", "/en/")

            complete_venues.append(venue)

    if not complete_venues:
        print(f"No complete venues found on page {page_number}.")
        return [], False

    print(f"Extracted {len(complete_venues)} venues from page {page_number}.")
    return complete_venues, False  # Continue crawling
