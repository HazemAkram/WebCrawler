# DeepSeek AI Web Crawler

A powerful, AI-powered web crawler that can extract product information from any e-commerce website and download associated PDF documents. Features advanced pagination handling, intelligent duplicate detection, and sophisticated PDF processing.

## 🚀 Features

### **Smart Pagination System**
- **Automatic Pattern Detection**: Detects pagination patterns across different websites
- **Multiple Pagination Types**: Supports page-based, offset-based, limit-based, and cursor-based pagination
- **Intelligent End Detection**: Automatically stops when no more results are found
- **Configurable Limits**: Set maximum pages and consecutive empty page limits

### **AI-Powered Extraction**
- **LLM-Based Extraction**: Uses Groq/DeepSeek R1 Distill Llama 70B for intelligent data extraction
- **Universal Compatibility**: Works with any website layout without custom configuration
- **Schema Validation**: Ensures extracted data meets required format

### **Advanced PDF Processing**
- **Automatic Download**: Downloads PDFs from product pages
- **Duplicate Prevention**: Prevents downloading identical files using content hashing
- **PDF Cleaning**: Removes QR codes and specific text with background color matching
- **Cover Page Addition**: Automatically adds custom cover pages

### **Robust Error Handling**
- **Retry Mechanism**: Automatically retries failed requests
- **Rate Limiting**: Respectful crawling with configurable delays
- **Session Management**: Efficient session handling for better performance

## 📁 Project Structure

```
.
├── main.py                    # Main entry point for the crawler
├── config.py                  # Configuration constants and pagination settings
├── cleaner.py                 # PDF processing and cleaning functionality
├── models/
│   └── venue.py              # Defines the Venue data model using Pydantic
├── utils/
│   ├── __init__.py           # Package marker for utils
│   ├── data_utils.py         # Utility functions for processing and saving data
│   └── scraper_utils.py      # Enhanced pagination and crawling utilities
├── requirements.txt          # Python package dependencies
├── .gitignore               # Git ignore file
└── README.MD                # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd deepseek-ai-web-crawler
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## 🚀 Usage

### **Basic Usage**

1. **Prepare your input file** (`stMotor.tsv`):
   ```
   https://example-store.com/products
   https://another-shop.com/catalog
   https://any-ecommerce-site.com/items
   ```

2. **Run the crawler**:
   ```bash
   python main.py
   ```

### **Advanced Configuration**

#### **Website-Specific Settings**

The crawler automatically detects website types and applies appropriate settings:

```python
# In config.py
WEBSITE_CONFIGS = {
    "your-website.com": {
        "type": "shopify",  # or "woocommerce", "magento", "manufacturer", "marketplace"
        "custom_patterns": ["page", "p"],
        "no_results_patterns": ["No products found"],
    }
}
```

#### **Pagination Configuration**

Configure pagination behavior for different website types:

```python
PAGINATION_CONFIGS = {
    "shopify": {
        "max_pages": 100,
        "max_consecutive_empty": 2,
        "delay_between_pages": (1, 3),
        "pagination_patterns": ["page", "p"],
    },
    "manufacturer": {
        "max_pages": 40,
        "max_consecutive_empty": 3,
        "delay_between_pages": (3, 6),
        "pagination_patterns": ["page", "offset", "start"],
    }
}
```

#### **Custom Pagination Handler**

For websites with unique pagination patterns:

```python
from utils.scraper_utils import PaginationHandler

# Create custom pagination handler
custom_handler = PaginationHandler()
custom_handler.pagination_patterns['custom'] = ['my_page_param', 'my_offset']

# Use in crawling
venues = await smart_pagination_crawl(
    crawler=crawler,
    base_url=url,
    pagination_handler=custom_handler,
    max_pages=50
)
```

## 🔧 Pagination System

### **Supported Pagination Types**

1. **Page-Based**: `?page=1`, `?p=2`, `?pg=3`
2. **Offset-Based**: `?offset=20`, `?start=40`, `?skip=60`
3. **Limit-Based**: `?limit=20&offset=40`
4. **Cursor-Based**: `?cursor=abc123`, `?after=xyz789`

### **Automatic Detection**

The crawler automatically detects pagination patterns by:
- Analyzing URL parameters
- Examining HTML pagination elements
- Checking for common pagination selectors
- Identifying "next" and "previous" buttons

### **End Detection**

The crawler intelligently stops when:
- "No results" messages are detected
- Pagination reaches the last page
- Consecutive empty pages exceed the limit
- Maximum page limit is reached

## 📊 Output

### **CSV Output**
The crawler generates a CSV file with extracted product data:
```csv
productName,productLink
Product 1,https://example.com/product1
Product 2,https://example.com/product2
```

### **PDF Downloads**
PDFs are organized by product name:
```
pdfs/
├── Product Name 1/
│   ├── document1.pdf
│   ├── document1.pdf_cleaned.pdf
│   └── document2.pdf
└── Product Name 2/
    ├── catalog.pdf
    └── catalog.pdf_cleaned.pdf
```

## ⚙️ Configuration Options

### **Crawling Behavior**
```python
CRAWLING_SETTINGS = {
    "retry_failed_pages": True,
    "max_retries": 3,
    "retry_delay": (5, 15),
    "respect_robots_txt": True,
    "follow_redirects": True,
    "timeout": 30,
    "max_concurrent_requests": 1,
}
```

### **PDF Processing**
```python
PDF_SETTINGS = {
    "max_file_size_mb": 50,
    "allowed_extensions": [".pdf"],
    "skip_existing": True,
    "create_cleaned_versions": True,
    "add_cover_page": True,
    "remove_qr_codes": True,
    "remove_specific_text": True,
}
```

## 🔍 Troubleshooting

### **Common Issues**

1. **No products extracted**:
   - Check if the website uses JavaScript rendering
   - Verify the URL is accessible
   - Check for anti-bot protection

2. **Pagination not working**:
   - Review the pagination configuration
   - Check if the website uses a custom pagination system
   - Verify the "no results" detection patterns

3. **PDF downloads failing**:
   - Check file size limits
   - Verify PDF links are accessible
   - Check for authentication requirements

### **Debug Mode**

Enable verbose logging by setting `verbose=True` in the browser configuration:

```python
def get_browser_config() -> BrowserConfig:
    return BrowserConfig(
        browser_type="chromium",
        headless=False,  # Set to True for production
        verbose=True,    # Enable debug logging
        headers={"User-Agent": user_agent},
    )
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Crawl4AI**: For the powerful web crawling framework
- **Groq**: For the fast LLM inference
- **DeepSeek**: For the intelligent extraction model
