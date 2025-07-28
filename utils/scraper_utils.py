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
    Only creates a product folder if PDFs are actually found.
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

        # Check if any PDFs were found
        if not pdf_links:
            print(f"📭 No PDFs found on page for product: {product_name}")
            print(f"🔗 Product URL: {product_url}")
            return  # Exit early without creating any folders

        print(f"📄 Found {len(pdf_links)} PDF(s) for product: {product_name}")

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
                
                # # Check if this exact PDF URL has been downloaded before (global tracking)
                # if pdf_url in download_pdf_links.downloaded_pdfs:
                #     print(f"⏭️ Skipping duplicate PDF URL (previously downloaded): {filename}")
                #     continue
                
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
                            # if hasattr(download_pdf_links, 'content_hashes') and content_hash in download_pdf_links.content_hashes:
                            #     print(f"⏭️ Skipping duplicate content: {filename}")
                            #     download_pdf_links.downloaded_pdfs.add(pdf_url)
                            #     continue
                            
                            # Store content hash and download the file
                            if not hasattr(download_pdf_links, 'content_hashes'):
                                download_pdf_links.content_hashes = set()
                            download_pdf_links.content_hashes.add(content_hash)
                            
                            async with aiofiles.open(save_path, "wb") as f:
                                await f.write(content)
                            
                            # Mark this URL as downloaded
                            download_pdf_links.downloaded_pdfs.add(pdf_url)
                            
                            # here should we put the script of the cleaning the pdf ? 
                            # pdf_processing(search_text=product_name.lower(), file_path=save_path)
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
        'page', 'p', 'pg', 'page_num', 'page_number', 'pageNumber',
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
        query_params['p'] = [str(page_number)]
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
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    
    # Check for existing pagination parameters
    if any(param in query_params for param in ['page', 'p', 'pg', 'page_num', 'pageNumber']):
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
        viewport_width = 1080,  # Width of the browser viewport
        viewport_height = 720,  # Height of the browser viewport
        verbose=True,  # Enable verbose logging
        user_agent = user_agent,  # Custom headers to include
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
            # api_token = "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" # openai API key     
        ),
        schema=Venue.model_json_schema(),  # JSON schema of the data model
        extraction_type="schema",  # Type of extraction to perform
        instruction=(
            "You are given HTML content that has already been filtered to include only the main product listing elements from an industrial or e-commerce website. This content was selected using a CSS selector, so it should primarily contain product cards, tiles, or grid items.\n"
            "\n"
            "Your task is to extract all valid product entries from this filtered HTML. For each product, extract the following fields:\n"
            "- productName: The complete product name or title, exactly as displayed on the website.\n"
            "- productLink: The full, absolute URL to the product detail page, taken from the href attribute of an anchor tag within the product element.\n"
            "\n"
            "Extraction Rules:\n"
            "- Only include entries that have both a productName and a productLink.\n"
            "- Use the exact product name as shown in the HTML—do not paraphrase or guess.\n"
            "- For productLink, always use the value from the href attribute and convert relative URLs to absolute URLs using the website domain.\n"
            "- Do not include categories, collections, or non-product items.\n"
            "- Do not guess or invent missing data; only extract what is present in the HTML.\n"
            "- Output a list of dictionaries, each matching the required schema.\n"
            "- If the url is incomplete, complete the url with the domain"
            "Context:\n"
            "The HTML you receive is already focused on product elements, so you do not need to search the entire page—just extract structured product data from the provided content.\n"
            "Output a list of dictionaries, each matching the required schema.\n"

        ),
        input_format="markdown",  # Format of the input content
        verbose=False,  # Enable verbose logging
    )


async def check_no_results(
    crawler: AsyncWebCrawler,
    css_selector: str,
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
            target_elements = css_selector,
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
    
    button_selector = ".load-more-button"  # CSS selector for the "Load More" button

    js_commands = f"""
        console.log('[JS] Starting data extraction...');
        let allRowsData = [];
        const rowSelectors = '{", ".join(css_selector)}';
        const buttonSelector = '{button_selector}';
        const maxPages = 3;
        let currentPage = 1;

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

        // Always extract first page
        allRowsData.push({{
            page: currentPage,
            data: extractRows()
        }});
        console.log(`[JS] Extracted initial page with ${{allRowsData[0].data.length}} rows`);

        // Only attempt pagination if valid button selector exists
        if (buttonSelector && buttonSelector.trim() !== '') {{
            console.log('[JS] Pagination detected. Starting automatic pagination...');
            let nextButton = document.querySelector(buttonSelector);
            
            while (currentPage < maxPages && nextButton && nextButton.offsetParent !== null && !nextButton.disabled) {{
                // Click to load next page
                nextButton.click();
                console.log(`[JS] Clicked page ${{currentPage}}`);
                currentPage++;
                
                // Wait for new content to load
                await new Promise(r => setTimeout(r, 3000));
                
                // Extract new page data
                const newPageData = extractRows();
                allRowsData.push({{
                    page: currentPage,
                    data: newPageData
                }});
                console.log(`[JS] Extracted page ${{currentPage}} with ${{newPageData.length}} rows`);
                
                // Update button reference after DOM changes
                nextButton = document.querySelector(buttonSelector);
            }}
            console.log('[JS] Pagination complete');
        }} else {{
            console.log('[JS] No pagination button selector provided. Returning single page data.');
        }}

        return allRowsData;
    """



    # Debugging: Print the URL being fetched
    print(f"Fetching page {page_number} from URL: {url}")    


    # Check if "No Results Found" message is present
    no_results = await check_no_results(
        crawler,
        css_selector,
        url,
        session_id
    )

    if no_results:
        print(f"------------------------------------------------------------------------- 🏁 No results found on page {page_number}. Stopping pagination. from the first run !! -------------------------------------------------------------------------"    )
        return [], True  # No more results, signal to stop crawling

    # Fetch page content with the extraction strategy
    result = await crawler.arun(
        url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,  # Do not use cached data
            extraction_strategy=llm_strategy,  # Strategy for data extraction
            target_elements = css_selector,  # Target specific content on the page
            # excluded_tags = ['script', 'style', 'head', 'footer', 'header', 'aside'],  # Target specific content on the page (we don't use it now but maybe will use it as a tag selector in the future)
            session_id=session_id,  # Unique session ID for the crawl
            # js_code=js_commands,  # JavaScript to handle pagination
            simulate_user= True, 
            verbose=True,  # Enable verbose logging
        ),
    )

    print(Venue.model_json_schema())

    resultfile = "result.txt"
    with open(resultfile, 'w', encoding='utf-8') as file: 
        file.write(str(result))
    print(f"result value saved into {resultfile}")

    htmlstring = result.html
    filename = "html_file.html"
    with open(filename, "w",  encoding='utf-8') as file:
        file.write(htmlstring)
    print(f"result.html saved to {filename}")

    htmlstring = result.cleaned_html
    filename = "cleaned_html_file.html"
    with open(filename, "w",  encoding='utf-8') as file:
        file.write(htmlstring)
    print(f"result.html saved to {filename}")

    htmlstring = result.fit_html
    filename = 'fit_html.html'
    with open(filename, 'w', encoding='utf-8') as file: 
        file.write(htmlstring)


    if not (result.success and result.extracted_content):
        print(f"Error fetching page {page_number}: {result.error_message}")
        return [], False

    # Parse extracted content
    extracted_data = json.loads(result.extracted_content)
    print(type(extracted_data))
    if not extracted_data:
        print(f"No products found on page {page_number}.")
        return [], False

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

    # ### APPLYING LLM BASED FILTER TO FILTER THE VENUES FROM THE NOISES ###
    # print(f"🔍 Applying LLM filter to {len(complete_venues)} extracted items...")
    
    # # Apply LLM filtering to remove non-product items
    # filtered_venues = await filter_products_with_llm(
    #     crawler=crawler,
    #     complete_venues=complete_venues,
    #     current_url=url,
    #     session_id=f"{session_id}_filter"
    # )
    
    # # Update the complete_venues list with filtered results
    # complete_venues = filtered_venues
    
    # print(f"✅ After LLM filtering: {len(complete_venues)} real products remaining")
    
    # if not complete_venues:
    #     print(f"No real products found on page {page_number} after filtering.")
    #     return [], False

    return complete_venues, False  # Continue crawling

# async def filter_products_with_llm(
#     crawler: AsyncWebCrawler,
#     complete_venues: List[dict],
#     current_url: str,
#     session_id: str = "product_filter_session"
# ) -> List[dict]:
#     """
#     Uses LLM to filter the complete_venues list and remove non-product items.
    
#     Args:
#         crawler (AsyncWebCrawler): The web crawler instance
#         complete_venues (List[dict]): List of extracted items (products and non-products)
#         current_url (str): The current page URL for context
#         session_id (str): Session identifier for the crawler
        
#     Returns:
#         List[dict]: Filtered list containing only real products
#     """
    
#     if not complete_venues:
#         print("No venues to filter.")
#         return []
    
#     try:
#         print(f"🔍 Filtering {len(complete_venues)} items using LLM...")
        
#         # Initialize Groq client with error handling
#         try:
#             # Try different initialization methods
#             try:
#                 client = groq.Groq(
#                     api_key="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"
#                 )
#             except TypeError:
#                 # Try alternative initialization if the first fails
#                 client = groq.Groq()
#                 client.api_key = "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"
            
#             print("✅ Groq client initialized successfully")
#         except Exception as groq_error:
#             print(f"❌ Failed to initialize Groq client: {groq_error}")
#             print(f"Groq error type: {type(groq_error)}")
#             print("Returning original list without filtering.")
#             return complete_venues
        
#         # Prepare the prompt
#         prompt = f"""
# You are given a JSON array that contains a mix of entries. Each entry includes a productName and a productLink. 
# Your task is to carefully analyze each item and return only the entries that represent actual physical products, such as specific devices or models.
# Include only entries that clearly refer to individual products (e.g., hardware items, identifiable models or SKUs).
# Exclude general categories, software, industry solutions, accessories, customer stories, or applications.
# Keep the original format of the JSON (same key names and structure).
# Output only the filtered list of actual products.
# Use clues like the URL structure, and naming patterns (e.g., specific product names vs. general terms).
# Return ONLY the JSON array:

# {json.dumps(complete_venues, indent=2)}
# """
        
#         # Call the LLM
#         chat_completion = client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             model="deepseek-r1-distill-llama-70b",
#             temperature=0.1,
#             max_tokens=4000
#         )
        
#         # Extract the response
#         llm_response = chat_completion.choices[0].message.content
        
#         # Parse the LLM response
#         try:
#             # Try to extract JSON from the response
#             import re
#             json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
#             if json_match:
#                 filtered_results = json.loads(json_match.group())
#             else:
#                 # If no JSON array found, try to parse the entire response
#                 filtered_results = json.loads(llm_response)
            
#             # Extract only the real products
#             real_products = []
#             for item in filtered_results:
#                 if item.get("isRealProduct", True):
#                     # Remove the filtering metadata and keep only product data
#                     product_data = {
#                         "productName": item["productName"],
#                         "productLink": item["productLink"]
#                     }
#                     real_products.append(product_data)
#                 else:
#                     print(f"❌ Filtered out: '{item['productName']}' - {item.get('reason', 'No reason provided')}")
            
#             print(f"✅ LLM filtering complete: {len(real_products)}/{len(complete_venues)} items kept as real products")
#             return real_products
            
#         except json.JSONDecodeError as e:
#             print(f"⚠️ Failed to parse LLM response: {e}")
#             print(f"LLM Response length: {len(llm_response)}")
#             print(f"LLM Response preview: {llm_response[:200]}...")
#             print(f"LLM Response end: ...{llm_response[-200:]}")
#             print("Returning original list without filtering.")
#             return complete_venues
            
#     except Exception as e:
#         print(f"⚠️ Error during LLM filtering: {e}")
#         print("Returning original list without filtering.")
#         return complete_venues