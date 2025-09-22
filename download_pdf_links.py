"""
Product-only runner

Reads a CSV containing columns: url, cat_name, pdf_selector, name_selector
and downloads PDFs for each product using the shared download pipeline.
Saves a CSV summary under CSVS/ with per-product saved file counts.
"""

import asyncio
import csv
import os
from datetime import datetime
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv
from utils.scraper_utils import (
    download_pdf_links,
    get_browser_config,
    get_pdf_llm_strategy,
    get_regex_strategy,
)

load_dotenv()


def _log(message: str):
    print(message)


def read_products_csv(csv_path: str):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("url") or "").strip()
            cat_name = (row.get("cat_name") or "Uncategorized").strip()
            pdf_selector = (row.get("pdf_selector") or "").strip()
            name_selector = (row.get("name_selector") or "").strip()
            selectors = [s.strip() for s in pdf_selector.split("|") if s.strip()]
            if name_selector:
                selectors.append(name_selector)
            if not url or not selectors:
                continue
            rows.append({
                "url": url,
                "cat_name": cat_name or "Uncategorized",
                "selectors": selectors,
            })
    return rows


async def run_products(csv_path: str, output_folder: str = "output"):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required in environment")

    model = "groq/llama-3.1-8b-instant"
    browser_config = get_browser_config()
    pdf_llm_strategy = get_pdf_llm_strategy(api_key=api_key, model=model)
    regex_strategy = get_regex_strategy()

    items = read_products_csv(csv_path)
    if not items:
        _log("No valid rows found in products CSV")
        return

    os.makedirs("CSVS", exist_ok=True)
    summaries = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for idx, item in enumerate(items, start=1):
            url = item["url"]
            selectors = item["selectors"]
            cat_name = item["cat_name"]
            parsed = urlparse(url)
            domain = parsed.netloc
            session_id = f"products_{idx}"
            _log(f"[{idx}/{len(items)}] Processing: {url}")
            try:
                summary = await download_pdf_links(
                    crawler=crawler,
                    product_url=url,
                    output_folder=output_folder,
                    pdf_selector=selectors,
                    session_id=session_id,
                    regex_strategy=regex_strategy,
                    domain_name=domain,
                    pdf_llm_strategy=pdf_llm_strategy,
                    api_key=api_key,
                    cat_name=cat_name,
                )
                if summary:
                    summaries.append({
                        "productLink": summary.get("productLink"),
                        "productName": summary.get("productName"),
                        "category": summary.get("category"),
                        "saved_count": summary.get("saved_count"),
                    })
            except Exception as e:
                _log(f"Error processing {url}: {e}")
            await asyncio.sleep(1)

    # Write summary CSV
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join("CSVS", f"products_downloaded_{ts}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["productLink","productName","category","saved_count"]) 
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)
        _log(f"Summary saved: {out_csv}")
    except Exception as e:
        _log(f"Failed to write summary CSV: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download product PDFs from a CSV of product URLs")
    parser.add_argument("--csv", required=True, help="Path to CSV with columns: url, cat_name, pdf_selector, name_selector")
    parser.add_argument("--output", default="output", help="Output folder")
    args = parser.parse_args()

    asyncio.run(run_products(args.csv, args.output))


if __name__ == "__main__":
    main()