import asyncio
import json

from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
)

# Assuming these are correctly implemented elsewhere
from models.venue import Venue
from config import REQUIRED_KEYS
from utils.data_utils import is_complete_venue, is_duplicate_venue
from utils.scraper_utils import get_browser_config, get_llm_strategy

async def main():
    llm_strategy = get_llm_strategy()
    browser_config = get_browser_config()
    button_selector = ".paginate_button.page-item.next > a"
    elemnts = ["tr.odd", "tr.even"]
    page_url = "https://www.wat.com.tr/urunler/3-fazli-aluminyum-govdeli-motorlar"

   
    js_commands = f"""
            console.log('[JS] Starting automatic pagination handling...');
            let allRowsData = [];
            let currentPage = 1;
            const maxClicks = 70;


            async function clickAndExtract() {{
                while (currentPage <= maxClicks) {{
                    const button = document.querySelector('{button_selector}');
                    
                    // Break conditions
                    if (!button || button.offsetParent === null || button.disabled) {{
                        console.log('[JS] Stopping: No more pages');
                        return allRowsData;
                    }}
                    
                    // Extract current page data BEFORE clicking next
                    const pageData = [];
                    const rows = document.querySelectorAll('{", ".join(elemnts)}');
                    
                    rows.forEach(row => {{
                        pageData.push({{
                            html: row.outerHTML
                        }});
                    }});
                    
                    allRowsData.push({{
                        page: currentPage,
                        data: pageData
                    }});
                    
                    // Click to load next page
                    button.click();
                    console.log(`[JS] Clicked page ${{currentPage}}`);
                    currentPage++;
                    
                    // Wait for new content
                    await new Promise(r => setTimeout(r, 3000));
                }}
                return allRowsData;
            }}

            return await clickAndExtract();
            """


    async with AsyncWebCrawler(config=browser_config) as crawler:
        required_keys = REQUIRED_KEYS
        complete_venues = []
        seen_names = set()
        page_number = 1       

        try:
            # Run with automatic pagination handling
            results = await crawler.arun(
                url=page_url,
                config=CrawlerRunConfig(
                    extraction_strategy=llm_strategy,
                    cache_mode=CacheMode.BYPASS,
                    target_elements = elemnts,
                    scan_full_page=True,
                    remove_overlay_elements=True,
                    page_timeout=30000,  # 3 minutes timeout
                    session_id="pepperl_fuchs_session",
                    js_code=js_commands
                )
            )

            
            filename = "result.txt"
            with open(filename, 'w', encoding = 'utf-8') as file : 
                file.write(str(results))
            print(f"JavaScript result saved to {filename}")

            jsstring = results.js_execution_result
            filename = "js_execution_result.txt"
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(str(jsstring))

            htmlstring = results.html
            filename = "html_file.html"
            with open(filename, "w",  encoding='utf-8') as file:
                file.write(htmlstring)
            print(f"result.html saved to {filename}")


            # Process extracted data
            if not results.extracted_content:
                print("No content extracted")
                return
                
            extracted_data = json.loads(results.extracted_content)
            print(f"Extracted {len(extracted_data)} products after pagination")
            
            # Process products
            new_products = 0
            for product in extracted_data:
                print(f"processing product: {product}")
                # Clean error field
                if product.get("error") is False:
                    product.pop("error", None)

                # Validate and deduplicate
                if not is_complete_venue(product, required_keys):
                    continue
                    
                if is_duplicate_venue(product["productLink"], seen_names):
                    print(f"Duplicate: {product['productName']}")
                    continue
                
                # Fix URL formatting
                if "productLink" in product:
                    product["productLink"] = product["productLink"].replace("/en/en/", "/en/")
                
                seen_names.add(product["productLink"])
                complete_venues.append(product)
                new_products += 1
            
            print(f"Added {new_products} unique products")
            print(f"Total venues collected: {len(complete_venues)}")
            
            # Here you would typically save/return complete_venues
            page_number += 1
        except Exception as e:
            print(f"Error during crawling: {str(e)}")
            return

if __name__ == "__main__":
    asyncio.run(main())