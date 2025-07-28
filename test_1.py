import asyncio
import json

from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    LLMExtractionStrategy,
)

# Assuming these are correctly implemented elsewhere
from models.venue import Venue
from config import REQUIRED_KEYS
from utils.data_utils import is_complete_venue, is_duplicate_venue
from utils.scraper_utils import get_browser_config, get_llm_strategy, download_pdf_links, get_regex_strategy
from typing import List, Set, Tuple


async def fetch_and_processing_page(
                                    crawler : AsyncWebCrawler,
                                    page_url : str,
                                    llm_strategy : LLMExtractionStrategy,
                                    button_selector : list,
                                    elements
                                    ) -> Tuple[List[dict], bool]:
    


   
    js_commands = f"""
        console.log('[JS] Starting data extraction...');
        let allRowsData = [];
        const rowSelectors = '{", ".join(elements)}';
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

    
    required_keys = REQUIRED_KEYS
    complete_venues = []
    seen_names = set()

    try:
        # Run with automatic pagination handling
        results = await crawler.arun(
            url=page_url,
            config=CrawlerRunConfig(
                extraction_strategy=None,
                cache_mode=CacheMode.BYPASS,
                target_elements = elements,
                scan_full_page=True,
                remove_overlay_elements=True,
                page_timeout=30000,  # 3 minutes timeout
                session_id="pepperl_fuchs_session",
                js_code=js_commands
            )
        )

        jsstring = results.js_execution_result
        filename = "js_execution_result.txt"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(str(jsstring))

        

        # Process extracted data
        if not results.js_execution_result:
            print("No content extracted")
            return [], False
        
        js_extracted_content = results.js_execution_result['results'][0]

        for items in js_extracted_content:
            print(f"processing page: {items['page']}, data length: {len(items['data'])}")
            counter = 1 

            products_string = str() 
            for item in items['data']: 
                products_string += f"\n{item['html']}"
                counter += 1

            seesion_id = f"popperl_fuchs_session_{items['page']}"
            raw_html_url = f"raw:\n{page_url}<div>\n{products_string}\n</div>"
            result = await crawler.arun(
                url = raw_html_url,
                config = CrawlerRunConfig(
                    extraction_strategy=llm_strategy,    
                    session_id = seesion_id
                    )
                )

            if not result.extracted_content:
                print(f"\tNo content extracted for item {counter} on page {items['page']}")
                return [], False
            
            extracted_content= json.loads(result.extracted_content)
            print(f"Extracted {len(extracted_content)}, product from page {items['page']}")
                
            new_products = 0 
            for product in extracted_content:
                print(f"\tprocessing product: {product}")
                if product.get("error") is False: 
                    product.pop("error", None)

                if not is_complete_venue(product, required_keys): 
                    continue
                
                if is_duplicate_venue(product["productLink"], seen_names):
                    print(f"\tDuplicate: {product['productName']}")
                    continue
                
                if "productLink" in product: 
                    product["productLink"] = product["productLink"].replace("/en/en/", "/en/")

                
                seen_names.add(product["productLink"])
                complete_venues.append(product)
                new_products += 1

            print(f"\tAdded {new_products} unique products")
            print(f"\tTotal products collected {len(complete_venues)}")
           

        if not complete_venues: 
            print("\tNo product to scrape")
            return [], False
        
        return complete_venues, True
                

    except Exception as e:
        print(f"Error during crawling: {str(e)}")
        return
        


async def main():

    llm_strategy = get_llm_strategy()
    regex_strategy = get_regex_strategy()
    browser_config = get_browser_config()
    button_selector = "" #".paginate_button.page-item.next > a"
    elements = ["td.product-name"]
    session_id = 'pdf_downloads'
    page_url = "https://omegamotor.com.tr/en/product/?rated_output=1;50;2;20;3;4;5;50;7;50;10;0;15;0;18;50;22;0;30;0;45;0;55;0&product_code=&rated_speed=&efficiency_class=IE3&body_material=Al&mountig=&min_rated_voltage=230;400&body_size=80;90;110;112;132;160;180;200;225&pol_number=2;4;6;8&service_factor=&view=1&page=0"


    async with AsyncWebCrawler(config = browser_config) as crawler:  
        products, no_results = await fetch_and_processing_page(
            crawler, 
            page_url,
            llm_strategy,
            button_selector,
            elements,
        )

        if no_results or not products: 
            print("🏁 Stopping pagination - no more results")

        
        for product in products :
            session_id = f"{session_id}_{product['productName']}"
            await download_pdf_links(
                crawler,
                product['productLink'], 
                product['productName'], 
                'test_pdf', 
                session_id,
                regex_strategy
            )

        


if __name__ == "__main__":
    asyncio.run(main())