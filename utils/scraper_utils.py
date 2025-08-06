"""
DeepSeek AI Web Crawler
Copyright (c) 2025 Ayaz Mensyoğlu

This file is part of the DeepSeek AI Web Crawler project.
Licensed under the Apache License, Version 2.0.
See NOTICE file for additional terms and conditions.
"""


import json
import os
import hashlib


import aiofiles
import aiohttp

from typing import List, Set, Tuple
from fake_useragent import UserAgent
from bs4 import BeautifulSoup

from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, urljoin

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


def generate_pdf_filename(pdf_url: str, product_name: str) -> str:
    """
    Generate an appropriate filename for a PDF URL, handling extensionless URLs.
    
    Args:
        pdf_url (str): The URL of the PDF file
        product_name (str): The name of the product for fallback naming
        
    Returns:
        str: A sanitized filename with .pdf extension
    """
    
    # Parse the URL
    parsed_url = urlparse(pdf_url)
    path = parsed_url.path
    query = parsed_url.query
    
    # Try to extract filename from URL path
    if path and path != '/':
        # Get the last part of the path
        path_parts = [part for part in path.split('/') if part]
        if path_parts:
            filename = path_parts[-1]
            # Remove any existing extension
            if '.' in filename:
                filename = filename.rsplit('.', 1)[0]
            # Add .pdf extension
            filename = f"{filename}.pdf"
        else:
            # No meaningful path, use query parameters or fallback
            filename = generate_filename_from_query(query, product_name)
    else:
        # No path, use query parameters or fallback
        filename = generate_filename_from_query(query, product_name)
    
    # Sanitize the filename
    filename = sanitize_filename(filename)
    
    return filename


def generate_filename_from_query(query: str, product_name: str) -> str:
    """
    Generate filename from URL query parameters or product name.
    
    Args:
        query (str): URL query string
        product_name (str): Product name for fallback
        
    Returns:
        str: Generated filename
    """
    if query:
        # Try to extract meaningful parameters
        params = parse_qs(query)
        
        # Look for common ID parameters
        for param_name in ['id', 'file_id', 'doc_id', 'download_id']:
            if param_name in params:
                return f"{param_name}_{params[param_name][0]}.pdf"
        
        # Look for type parameters
        for param_name in ['type', 'file_type', 'doc_type']:
            if param_name in params:
                return f"document_{params[param_name][0]}.pdf"
        
        # Use first parameter as fallback
        first_param = list(params.keys())[0]
        first_value = params[first_param][0]
        return f"{first_param}_{first_value}.pdf"
    
    # Ultimate fallback: use product name
    return f"{sanitize_folder_name(product_name)}.pdf"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to be safe for filesystem use.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = r'[<>:"|?*\x00-\x1f]'
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Replace multiple consecutive underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Limit length to prevent filesystem issues
    if len(sanitized) > 200:  # Leave room for path
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:200-len(ext)-1] + (f'.{ext}' if ext else '')
    
    # Ensure the name is not empty
    if not sanitized:
        sanitized = "unnamed_document.pdf"
    
    return sanitized


# https://www.ors.com.tr/en/tek-sirali-sabit-bilyali-rulmanlar
async def download_pdf_links(
        crawler: AsyncWebCrawler, 
        product_url: str, 
        product_name: str,
        output_folder: str, 
        session_id="pdf_download_session", 
        regex_strategy: RegexExtractionStrategy = None , 
        domain_name: str = None,
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
        soup = BeautifulSoup(html, "html.parser")

        # page_content = response.content

        # Extract all <a> href links
        pdf_links = []
        seen_pdf_urls_in_page = set()  # Track PDF URLs found on this specific page

        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(product_url, href)
            
            # Enhanced PDF detection - check multiple indicators
            is_pdf_link = False
            
            # 1. Check for .pdf extension
            if href.lower().endswith(".pdf"):
                is_pdf_link = True
            # 2. Check for PDF-related keywords in URL
            elif any(keyword in href.lower() for keyword in ['pdf', 'download', 'document', 'manual', 'catalog', 'datasheet', 'brochure', 'specification', 'technical', 'data', 'sheet', 'guide', 'instruction']):
                is_pdf_link = True

            # 3. Check for PDF-related keywords in link text
            elif a.get_text().strip():
                link_text = a.get_text().strip().lower()
                if any(keyword in link_text for keyword in ['pdf', 'download', 'document', 'manual', 'catalog', 'datasheet', 'brochure', 'specification', 'technical', 'data sheet', 'user guide', 'instruction', 'installation', 'operation']):
                    is_pdf_link = True

            # 4. Check for common PDF download patterns
            elif any(pattern in href.lower() for pattern in ['/download', '/file', '/doc', '/attachment', '/media', '/assets', '/files', '/documents']):
                is_pdf_link = True

            # 5. Check for industrial/manufacturing specific patterns
            elif any(pattern in href.lower() for pattern in ['/technical', '/specs', '/specifications', '/data', '/info', '/details']):
                is_pdf_link = True
            
            if is_pdf_link:
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


        # Download each PDF with duplicate checking (SSL verification disabled)
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            for pdf_url in pdf_links:
                # Enhanced filename generation for extensionless URLs
                filename = generate_pdf_filename(pdf_url, product_name)
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
                            # Validate content type for PDF files
                            content_type = resp.headers.get('content-type', '').lower()
                            is_pdf_content = (
                                'application/pdf' in content_type or
                                'pdf' in content_type or
                                content_type.startswith('application/octet-stream') or
                                content_type.startswith('binary/')
                            )
                            
                            # Read the content to check for duplicates
                            content = await resp.read()
                            
                            # Additional validation: check if content starts with PDF magic bytes
                            if not is_pdf_content and len(content) >= 4:
                                pdf_magic_bytes = b'%PDF'
                                if not content.startswith(pdf_magic_bytes):
                                    print(f"⚠️ Skipping non-PDF content from {pdf_url} (Content-Type: {content_type})")
                                    continue
                            
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
                            
                            # Ensure filename has .pdf extension if it doesn't already
                            if not filename.lower().endswith('.pdf'):
                                filename += '.pdf'
                                save_path = os.path.join(productPath, filename)
                            
                            async with aiofiles.open(save_path, "wb") as f:
                                await f.write(content)
                            
                            # Mark this URL as downloaded
                            download_pdf_links.downloaded_pdfs.add(pdf_url)


                            search_text = [
                                # English     Türk     German
                                'Address', 'Adres', 'Adresse', 
                                'Telephone', 'Telefon',
                                'Fax', 'Faks', 
                                'Email', 'e-posta','e-mail', 'eposta',
                                domain_name, f"www.{domain_name}"
                                #product_name.lower()  # Also include product name for removal
                                ]
                            # here should we put the script of the cleaning the pdf ? 
                            pdf_processing(search_text_list=search_text, file_path=save_path)
                            print(f"📄 Downloaded PDF: {save_path}")
                        else:
                            print(f"❌ Failed to download: {pdf_url} (Status: {resp.status})")
                except Exception as e:
                    print(f"❌ Error downloading {pdf_url}: {e}")

    except Exception as e:
        print(f"⚠️ Error During processing  {product_url} pdf : {e}")

def get_page_number(base_url: str): 
    
    try:
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

        pagintaion_type = ""
        for param in query_params: 
            for rparam in pagination_params_to_remove: 
                if param == rparam: 
                    pagintaion_type = param

        return int(query_params[pagintaion_type][0])
    except Exception as e: 
        
        if str(e) == "''": 
            print(f"⚠️ Error During Extracting Page Number URL IS NOT PAGINTABLE ")
            return None
        
        else : 
            print(f"⚠️ Error During Extracting Page Number : {e}")
            return None


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
    try: 
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
        pagination_type = ""
        for param in query_params: 
            for r_param in pagination_params_to_remove: 
                if param == r_param: 
                    pagination_type = param
                    break

        pagintaion_page = query_params[pagination_type]
        query_params.pop(pagination_type, None)

        # Calculate pagination values based on type
        if (
            pagination_type == "page" or 
            pagination_type == 'p' or 
            pagination_type == 'pg' or 
            pagination_type == 'page_num' or 
            pagination_type == 'page_number' or 
            pagination_type == 'pageNumber'
        ):
            query_params[pagination_type] = [str(page_number)]
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
        # else:
        #     # Default to page-based pagination
        #     query_params['page'] = [str(page_number)]
        
        # Reconstruct the URL
        new_query = urlencode(query_params, doseq=True)
        new_parsed = parsed._replace(query=new_query)
        
        return urlunparse(new_parsed)
    except Exception as e :

        if str(e)  == "''": 
            print(f"⚠️ Error During Appending page parameter URL IS NOT PAGINTABLE")
            return base_url
        else: 
            print(f"⚠️ Error during Append Page Parameter : {e}")
            return base_url


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
        headless=True,  # Whether to run in headless mode (no GUI)
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



def get_llm_strategy(api_key: str = None, model: str = "groq/deepseek-r1-distill-llama-70b") -> LLMExtractionStrategy:
    """
    Returns the configuration for the language model extraction strategy.
    
    Args:
        api_key (str): The API key for the LLM provider. If None, will try to get from environment.
        model (str): The LLM model to use. Defaults to "groq/deepseek-r1-distill-llama-70b".
    
    Returns:
        LLMExtractionStrategy: The settings for how to extract data using LLM.
    """
    # Use provided API key or fall back to environment variable
    if api_key is None:
        api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        raise ValueError("API key is required. Please provide an API key or set GROQ_API_KEY environment variable.")
    
    # https://docs.crawl4ai.com/api/strategies/#llmextractionstrategy
    return LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider = model, # LLM model to use
            api_token= api_key,  # API token for the LLM provider    
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
            session_id=session_id,  # Unique session ID for the crawl
            scan_full_page=True,
            remove_overlay_elements=True,
            page_timeout=30000,
            verbose=True,  # Enable verbose logging

        ),
    )

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

    return complete_venues, False  # Continue crawling

async def fetch_and_process_page_with_js(
    crawler: AsyncWebCrawler,
    page_url: str,
    llm_strategy: LLMExtractionStrategy,
    button_selector: str,
    elements: list,
    required_keys: list,
    seen_names: set,
) -> Tuple[list, bool]:
    """
    JS-based extraction for sites with dynamic pagination. Extracts product rows using JS, then applies LLM extraction per page.
    Returns (venues, no_results) just like fetch_and_process_page.
    """
    js_commands = f"""
        console.log('[JS] Starting data extraction...');
        let allRowsData = [];
        const rowSelectors = '{", ".join(elements)}';
        const buttonSelector = '{button_selector}';
        const maxPages = 2;
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

    complete_venues = []
    try:
        results = await crawler.arun(
            url=page_url,
            config=CrawlerRunConfig(
                extraction_strategy=None,
                cache_mode=CacheMode.BYPASS,
                target_elements=elements,
                scan_full_page=True,
                remove_overlay_elements=True,
                page_timeout=30000,
                session_id="js_extraction_session",
                js_code=js_commands,
            )
        )


        jsstring = results.js_execution_result
        filename = "js_execution_result.txt"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(str(jsstring))

        js_extracted_content = None
        if hasattr(results, 'js_execution_result') and results.js_execution_result:
            # Try to get the results from the JS execution
            if isinstance(results.js_execution_result, dict) and 'results' in results.js_execution_result:
                js_extracted_content = results.js_execution_result['results'][0]
            else:
                js_extracted_content = results.js_execution_result
        if not js_extracted_content:
            print("No content extracted via JS")
            return [], True
        for items in js_extracted_content:
            print(f"processing page: {items['page']}, data length: {len(items['data'])}")
            products_string = "".join([item['html'] for item in items['data']])
            session_id = f"js_extraction_session_{items['page']}"
            raw_html_url = f"raw:\n{page_url}<div>\n{products_string}\n</div>"
            result = await crawler.arun(
                url=raw_html_url,
                config=CrawlerRunConfig(
                    extraction_strategy=llm_strategy,
                    session_id=session_id
                )
            )
            if not result.extracted_content:
                print(f"\tNo content extracted for page {items['page']}")
                continue
            extracted_content = json.loads(result.extracted_content)
            print(f"Extracted {len(extracted_content)} products from page {items['page']}")
            new_products = 0
            for product in extracted_content:
                print(f"\tprocessing product: {product}")
                if product.get("error") is False:
                    product.pop("error", None)
                if not is_complete_venue(product, required_keys):
                    continue
                if is_duplicate_venue(product["productName"], seen_names):
                    print(f"\tDuplicate: {product['productName']}")
                    continue
                if "productLink" in product:
                    product["productLink"] = product["productLink"].replace("/en/en/", "/en/")
                seen_names.add(product["productName"])
                complete_venues.append(product)
                new_products += 1
            print(f"\tAdded {new_products} unique products")
        if not complete_venues:
            print("\tNo product to scrape")
            return [], True
        return complete_venues, False
    except Exception as e:
        print(f"Error during JS-based crawling: {str(e)}")
        return [], True
