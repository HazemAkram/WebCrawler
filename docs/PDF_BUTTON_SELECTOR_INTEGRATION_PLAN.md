# Integration Plan: Supporting `pdf_button_selector` for Dynamic PDF Downloads

---

## Introduction
Some product sites require you to click a button that triggers a PDF file download (rather than providing a direct PDF link). Our web crawler, built on Playwright (via Crawl4AI), already supports browser automation, so we can extend it to handle these situations reliably.

This plan guides you, step-by-step, to:
- Add a new column to `sites.csv`, so each site can specify a selector for a PDF download button (if needed)
- Update the code to read and pass this information through the pipeline
- Modify the crawling & PDF download logic to click such buttons automatically
- Provide robust error handling, thorough testing, and code/documentation updates

---

## 1. Update the CSV Configuration (`sites.csv`)
### 1.1 Add New Column
- Open `sites.csv` in your code editor or Excel.
- Add a new column at the end (or after `button_selector`) called: `pdf_button_selector`
- Example **header row**:

```csv
url,cat_name,css_selector,pdf_selector,name_selector,button_selector,pdf_button_selector
```

- For sites that require pressing a button to get the PDF, fill in the exact CSS selector for the download button.
- If not needed, just leave that cell empty for the respective row.

#### **Sample Row**
```csv
https://example.com,...,a.item-class,...,h1.title,,button.pagination,.download-btn
```
- Here, `.download-btn` is the selector for the PDF download button on the product page.

---

## 2. Update CSV Reading Logic
### 2.1 Locate Parsing Logic
- **File:** Usually `main.py` (function: `read_sites_from_csv`).
- Locate the current block where you extract values from each row (`row.get(...)` or `row[...]`).

### 2.2 Add `pdf_button_selector` Extraction
- After parsing the other selectors, add:

```python
pdf_button_selector = row.get("pdf_button_selector", "").strip()
```

### 2.3 Add to Site Dict
- Inside the `.append({...})` call, add:

```python
    "pdf_button_selector": pdf_button_selector,
```

- Full site dict example:

```python
sites.append({
    "url": row["url"],
    "cat_name": row.get("cat_name", "Uncategorized"),
    "css_selector": css_list,
    "pdf_selector": pdf_list,
    "button_selector": row.get("button_selector", ""),
    "pdf_button_selector": pdf_button_selector,  # NEW
})
```

---

## 3. Pass Config Through the Pipeline
### 3.1 Update Function Calls
- Anywhere you use a site config (such as when you pass parameters to a function for crawling or PDF downloading), make sure to also pass the new `pdf_button_selector` value.
- Example in `main.py`:

```python
await download_pdf_links(
    crawler=pdf_crawler,
    product_url=venue["productLink"],
    output_folder="output",
    pdf_selector=site["pdf_selector"],
    ...,
    pdf_button_selector=site.get("pdf_button_selector", ""),  # new
)
```

- **Update all related interfaces and utility functions as required.**

---

## 4. Update PDF Download Logic
### 4.1 Update Function Signature
- In `utils/scraper_utils.py`, find the `download_pdf_links` routine, and update its signature to accept the new argument:

```python
async def download_pdf_links(
    crawler: AsyncWebCrawler,
    product_url: str,
    output_folder: str,
    pdf_llm_strategy: LLMExtractionStrategy,
    pdf_selector: str | list,
    session_id="pdf_download_session",
    regex_strategy: RegexExtractionStrategy = None,
    domain_name: str = None,
    api_key: str = None,
    cat_name: str = "Uncategorized",
    pdf_button_selector: str = "",   # ADD THIS
):
```

### 4.2 Inject the Button Click
- Immediately after your code loads/navigates to the product or PDF page, add logic for clicking the button if a selector was provided.
- This should happen **before** searching for any links to download:

```python
if pdf_button_selector:
    try:
        await crawler.page.click(pdf_button_selector)
        await crawler.page.wait_for_timeout(2000)  # Wait for the file/link to become available (adjust as needed!)
        log_message(f"Clicked PDF download button: {pdf_button_selector}", "INFO")
    except Exception as e:
        log_message(f"Failed to click PDF download button ({pdf_button_selector}): {e}", "WARNING")
```
- Now continue with the usual logic to search for links and capture PDFs.

---

## 5. Error Handling
- Errors when clicking the button **should not crash** the process; they should be logged as warnings and the scraper should try to proceed (in case the PDF is also available by other means).
- Ensure your logging is informative and appears in log files or dashboard if available (see `log_message` usage above).

---

## 6. Download Event Handling (If Triggered As Download)
- If clicking the button causes the browser to start a download (rather than exposing a link), you may need to explicitly use Playwright's download event API to intercept/download the file. Example:

```python
if pdf_button_selector:
    async with crawler.page.expect_download() as download_info:
        await crawler.page.click(pdf_button_selector)
    download = await download_info.value
    save_path = os.path.join(output_folder, 'myfile.pdf')
    await download.save_as(save_path)
```
- Integrate such logic **if and only if** direct link scraping does not work after a button press on affected sites.

---

## 7. End-to-End Testing
### 7.1 Prepare CSV Rows
- At least one row with `pdf_button_selector` set to a valid selector that shows or triggers the PDF download.
- At least one row with it blank (for legacy/direct-link cases).

### 7.2 Manual Testing
- Run the crawler (locally, in dev mode).
- Confirm:
    - PDF downloads succeed for button-triggered cases.
    - Downloads *still* work for normal, already-supported sites.
    - Logs show attempted button clicks and any warnings as expected.

### 7.3 Troubleshooting Tips
- If nothing downloads: check the selector in browser devtools and update it if needed.
- If the file downloads to a random location: ensure the Playwright/Crawl4AI download directory is set up correctly.
- If multiple PDFs are present: confirm your selectors are correct in the CSV (target only the right button).
- If you get rate-limited by a website: increase rate-limit/timings in code (`await asyncio.sleep(...)`).

---

## 8. Update Documentation
- Update the `README.md`:
    - Explain what the new column does, with a CSV sample.
    - Offer a short example for developers/integrators.
- Document the new logic in any developer or API reference docs as needed.

---

## 9. Code Style & Type Checking
- Update all type hints, docstrings, etc. in changed functions.
- Run your project’s linter (e.g., `flake8`/`ruff`) and fix all issues.

---

## **Full `download_pdf_links` Example (after update)**
```python
async def download_pdf_links(
    crawler: AsyncWebCrawler,
    product_url: str,
    output_folder: str,
    pdf_llm_strategy: LLMExtractionStrategy,
    pdf_selector: str | list,
    session_id="pdf_download_session",
    regex_strategy: RegexExtractionStrategy = None,
    domain_name: str = None,
    api_key: str = None,
    cat_name: str = "Uncategorized",
    pdf_button_selector: str = "",   # <--- NEW!
):
    await crawler.page.goto(product_url)

    if pdf_button_selector:
        try:
            await crawler.page.click(pdf_button_selector)
            await crawler.page.wait_for_timeout(2000)  # Adjust for download appearance
            log_message(f"Clicked PDF button: {pdf_button_selector}", "INFO")
        except Exception as e:
            log_message(f"Failed to click PDF button ({pdf_button_selector}): {e}", "WARNING")

    # Continue your PDF-finding and downloading logic as before!
    ...
```

---

## **Checklist Before Marking Task As Complete**
- [ ] Code changes work for button-triggered PDF downloads
- [ ] Code changes do not break non-button PDF downloads
- [ ] CSV with new column works for all use cases
- [ ] Tests and logging confirm robust behavior
- [ ] Docs updated — usage can be understood by other devs

---