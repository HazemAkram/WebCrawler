import json
import os
import hashlib
import re
import ssl 
import certifi

import aiofiles
import aiohttp
from urllib.parse import urljoin, urlparse

from typing import List, Set, Tuple
from fake_useragent import UserAgent
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import re

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


from models.venue import Venue
from utils.data_utils import is_complete_venue, is_duplicate_venue



load_dotenv()

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
    sanitized = product_name.replace('\\', '_').replace('/', '_')
    
    # Replace other invalid characters
    invalid_chars = r'[<>:"|?*]'
    sanitized = re.sub(invalid_chars, '_', sanitized)
    
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

# https://www.ors.com.tr/en/tek-sirali-sabit-bilyali-rulmanlar
async def download_pdf_links(
        crawler: AsyncWebCrawler, 
        product_url: str, 
        product_name: str,
        output_folder: str, 
        session_id="pdf_download_session", 
        regex_strategy: RegexExtractionStrategy = None 
        ):
    
    """
    Opens the given product page, searches for any PDF links, and downloads them.
    Prevents downloading duplicate PDFs by checking existing files.
    """

    # Global set to track downloaded PDFs across all products
    if not hasattr(download_pdf_links, 'downloaded_pdfs'):
        download_pdf_links.downloaded_pdfs = set()

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
        seen_pdf_urls_in_page = set()  # Track PDF URLs found on this specific page

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                full_url = urljoin(product_url, href)
                # Check for duplicates within this page
                if full_url not in seen_pdf_urls_in_page:
                    pdf_links.append(full_url)
                    seen_pdf_urls_in_page.add(full_url)
                else:
                    print(f"⏭️ Skipping duplicate PDF URL found on same page: {os.path.basename(urlparse(full_url).path)}")

        # Create the download folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        productPath = output_folder + f'/{sanitize_folder_name(product_name)}' 
        if not os.path.exists(productPath):
            os.makedirs(productPath)


        # disable SSL verification for the session
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Download each PDF with duplicate checking
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            for pdf_url in pdf_links:
                filename = os.path.basename(urlparse(pdf_url).path)
                save_path = os.path.join(productPath, filename)
                
                # Check if this exact PDF URL has been downloaded before (global tracking)
                if pdf_url in download_pdf_links.downloaded_pdfs:
                    print(f"⏭️ Skipping duplicate PDF URL (previously downloaded): {filename}")
                    continue
                
                # # Check if file already exists in the product folder
                # if os.path.exists(save_path):
                #     print(f"⏭️ File already exists: {save_path}")
                #     download_pdf_links.downloaded_pdfs.add(pdf_url)
                #     continue
                
                try:
                    async with session.get(pdf_url) as resp:
                        if resp.status == 200:
                            # Read the content to check for duplicates
                            content = await resp.read()
                            
                            # Check if content is identical to any existing PDF
                            content_hash = hashlib.md5(content).hexdigest()
                            if hasattr(download_pdf_links, 'content_hashes') and content_hash in download_pdf_links.content_hashes:
                                print(f"⏭️ Skipping duplicate content: {filename}")
                                download_pdf_links.downloaded_pdfs.add(pdf_url)
                                continue
                            
                            # Store content hash and download the file
                            if not hasattr(download_pdf_links, 'content_hashes'):
                                download_pdf_links.content_hashes = set()
                            download_pdf_links.content_hashes.add(content_hash)
                            
                            async with aiofiles.open(save_path, "wb") as f:
                                await f.write(content)
                            
                            # Mark this URL as downloaded
                            download_pdf_links.downloaded_pdfs.add(pdf_url)
                            
                            # here should we put the script of the cleaning the pdf ? 
                            pdf_processing(search_text=product_name.lower(), file_path=save_path)
                            print(f"📄 Downloaded PDF: {save_path}")
                        else:
                            print(f"❌ Failed to download: {pdf_url}")
                except Exception as e:
                    print(f"❌ Error downloading {pdf_url}: {e}")

    except Exception as e:
        print(f"⚠️ Error processing {product_url}: {e}")

def append_page_param(base_url: str, page_number: int, pagination_type: str = "auto") -> str:
    """
    Enhanced pagination parameter handler that supports multiple pagination patterns.
    
    Args:
        base_url (str): The base URL to append pagination to
        page_number (int): The page number to navigate to
        pagination_type (str): Type of pagination to use. Options:
            - "auto": Automatically detect pagination type from URL
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
    
    # Parse the URL
    parsed = urlparse(base_url)
    query_params = parse_qs(parsed.query)
    
    # Remove any existing pagination parameters
    pagination_params_to_remove = [
        'page', 'p', 'pg', 'page_num', 'page_number',
        'offset', 'start', 'skip', 'from',
        'limit', 'size', 'per_page', 'items_per_page',
        'cursor', 'after', 'before', 'next', 'prev',
        'page_id', 'pageid', 'pageno', 'pagenum'
    ]
    
    for param in pagination_params_to_remove:
        query_params.pop(param, None)
    
    # Determine pagination type if auto
    if pagination_type == "auto":
        pagination_type = _detect_pagination_type(base_url)
    
    # Calculate pagination values based on type
    if pagination_type == "page":
        query_params['page'] = [str(page_number)]
    elif pagination_type == "offset":
        query_params['offset'] = [str((page_number - 1) * 20)]  # Assuming 20 items per page
    elif pagination_type == "start":
        query_params['start'] = [str((page_number - 1) * 20)]
    elif pagination_type == "skip":
        query_params['skip'] = [str((page_number - 1) * 20)]
    elif pagination_type == "limit_offset":
        query_params['limit'] = ['20']
        query_params['offset'] = [str((page_number - 1) * 20)]
    elif pagination_type == "cursor":
        # For cursor-based, we'll use a simple numeric cursor
        # In real scenarios, you might need to get the actual cursor from previous page
        query_params['cursor'] = [str(page_number * 20)]
    elif pagination_type == "after":
        query_params['after'] = [str(page_number * 20)]
    elif pagination_type == "before":
        query_params['before'] = [str(page_number * 20)]
    else:
        # Default to page-based pagination
        query_params['page'] = [str(page_number)]
    
    # Reconstruct the URL
    new_query = urlencode(query_params, doseq=True)
    new_parsed = parsed._replace(query=new_query)
    
    return urlunparse(new_parsed)


def _detect_pagination_type(url: str) -> str:
    """
    Automatically detect the pagination type from URL patterns.
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        str: Detected pagination type
    """
    from urllib.parse import urlparse, parse_qs
    
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    
    # Check for existing pagination parameters
    if any(param in query_params for param in ['page', 'p', 'pg', 'page_num']):
        return "page"
    elif any(param in query_params for param in ['offset', 'start', 'skip']):
        return "offset"
    elif 'limit' in query_params:
        return "limit_offset"
    elif any(param in query_params for param in ['cursor', 'after', 'before']):
        return "cursor"
    
    # Check URL path patterns
    path = parsed.path.lower()
    if any(pattern in path for pattern in ['/page/', '/p/', '/pg/']):
        return "page"
    elif any(pattern in path for pattern in ['/offset/', '/start/', '/skip/']):
        return "offset"
    
    # Check domain-specific patterns
    domain = parsed.netloc.lower()
    if any(platform in domain for platform in ['shopify', 'woocommerce', 'magento']):
        return "page"
    elif any(platform in domain for platform in ['api.', 'rest.', 'graphql']):
        return "offset"
    
    # Default to page-based pagination
    return "page"


def create_pagination_url(base_url: str, page_number: int, items_per_page: int = 20, 
                         pagination_type: str = "auto", custom_params: dict = None) -> str:
    """
    Advanced pagination URL creator with support for custom parameters and items per page.
    
    Args:
        base_url (str): The base URL
        page_number (int): The page number (1-based)
        items_per_page (int): Number of items per page
        pagination_type (str): Type of pagination to use
        custom_params (dict): Additional custom parameters to include
        
    Returns:
        str: Complete pagination URL
    """
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    
    # Parse the URL
    parsed = urlparse(base_url)
    query_params = parse_qs(parsed.query)
    
    # Add custom parameters if provided
    if custom_params:
        for key, value in custom_params.items():
            query_params[key] = [str(value)]
    
    # Calculate offset for offset-based pagination
    offset = (page_number - 1) * items_per_page
    
    # Remove existing pagination parameters
    pagination_params_to_remove = [
        'page', 'p', 'pg', 'page_num', 'page_number',
        'offset', 'start', 'skip', 'from',
        'limit', 'size', 'per_page', 'items_per_page',
        'cursor', 'after', 'before', 'next', 'prev'
    ]
    
    for param in pagination_params_to_remove:
        query_params.pop(param, None)
    
    # Apply pagination based on type
    if pagination_type == "auto":
        pagination_type = _detect_pagination_type(base_url)
    
    if pagination_type == "page":
        query_params['page'] = [str(page_number)]
    elif pagination_type == "offset":
        query_params['offset'] = [str(offset)]
    elif pagination_type == "start":
        query_params['start'] = [str(offset)]
    elif pagination_type == "skip":
        query_params['skip'] = [str(offset)]
    elif pagination_type == "limit_offset":
        query_params['limit'] = [str(items_per_page)]
        query_params['offset'] = [str(offset)]
    elif pagination_type == "size_offset":
        query_params['size'] = [str(items_per_page)]
        query_params['offset'] = [str(offset)]
    elif pagination_type == "per_page":
        query_params['per_page'] = [str(items_per_page)]
        query_params['page'] = [str(page_number)]
    elif pagination_type == "items_per_page":
        query_params['items_per_page'] = [str(items_per_page)]
        query_params['page'] = [str(page_number)]
    else:
        # Default to page-based
        query_params['page'] = [str(page_number)]
    
    # Reconstruct URL
    new_query = urlencode(query_params, doseq=True)
    new_parsed = parsed._replace(query=new_query)
    
    return urlunparse(new_parsed)


def get_pagination_info(url: str) -> dict:
    """
    Extract pagination information from a URL.
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        dict: Dictionary containing pagination information
    """
    from urllib.parse import urlparse, parse_qs
    
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    
    info = {
        'current_page': 1,
        'items_per_page': 20,
        'offset': 0,
        'pagination_type': 'unknown',
        'has_pagination': False
    }
    
    # Detect pagination type and extract values
    if 'page' in query_params:
        info['pagination_type'] = 'page'
        info['current_page'] = int(query_params['page'][0])
        info['has_pagination'] = True
    elif 'offset' in query_params:
        info['pagination_type'] = 'offset'
        info['offset'] = int(query_params['offset'][0])
        info['current_page'] = (info['offset'] // info['items_per_page']) + 1
        info['has_pagination'] = True
    elif 'limit' in query_params and 'offset' in query_params:
        info['pagination_type'] = 'limit_offset'
        info['items_per_page'] = int(query_params['limit'][0])
        info['offset'] = int(query_params['offset'][0])
        info['current_page'] = (info['offset'] // info['items_per_page']) + 1
        info['has_pagination'] = True
    
    return info

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
            "Extract ALL products from this webpage. Focus on product grids, cards, and catalog items.\n\n"
            "EXTRACT:\n"
            "- productName: Complete product name/title\n"
            "- productLink: Full URL to product detail page\n\n"
            "IGNORE: Navigation menus, footers, sidebars, banners, ads, pagination\n\n"
            "RULES:\n"
            "- Only include products from the same category with both name and link\n"
            "- Use exact names as displayed\n"
            "- Convert relative URLs to absolute\n"
            "- No guessing missing data\n"
            "- Include each product only once\n\n"
            "Look for: Product cards, clickable product names, table listings, catalog items"
            "Output a list of dictionaries following the required schema."

        ),
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
            css_selector= "div",  # Target specific content on the page 
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
