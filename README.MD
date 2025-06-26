# DeepSeek AI Web Crawler

A powerful, AI-powered web crawler that can extract product information from any e-commerce website and download associated PDF documents. Features enhanced pagination handling, intelligent duplicate detection, and sophisticated PDF processing.

## 🚀 Features

### **Enhanced Pagination System**
- **Universal Compatibility**: Works with any website or API pagination pattern
- **Automatic Detection**: Detects pagination type from URL patterns (page-based, offset-based, limit-offset, cursor-based)
- **Intelligent Parameter Management**: Preserves existing URL parameters while managing pagination
- **Smart End Detection**: Automatically stops when no more results are found

### **AI-Powered Extraction**
- **LLM-Based Extraction**: Uses Groq/DeepSeek R1 Distill Llama 70B for intelligent data extraction
- **Universal Compatibility**: Works with any website layout without custom configuration
- **Schema Validation**: Ensures extracted data meets required format using Pydantic models

### **Advanced PDF Processing**
- **Automatic Download**: Downloads PDFs from product pages
- **Duplicate Prevention**: Prevents downloading identical files using content hashing
- **PDF Cleaning**: Removes QR codes and specific text with background color matching
- **Cover Page Addition**: Automatically adds custom cover pages

### **Robust Error Handling**
- **Graceful Degradation**: Continues crawling if individual pages fail
- **Rate Limiting**: Respectful crawling with configurable delays
- **Session Management**: Efficient session handling for better performance

## 📁 Project Structure

```
.
├── main.py                    # Main entry point for the crawler
├── config.py                  # Configuration constants
├── cleaner.py                 # PDF processing and cleaning functionality
├── models/
│   └── venue.py              # Defines the Venue data model using Pydantic
├── utils/
│   ├── __init__.py           # Package marker for utils
│   ├── data_utils.py         # Utility functions for processing and saving data
│   └── scraper_utils.py      # Enhanced pagination and crawling utilities
├── requirements.txt          # Python package dependencies
├── .gitignore               # Git ignore file
├── stMotor.tsv              # Input file with URLs to crawl
├── output.csv               # Output file with extracted products
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

4. **Install additional dependencies for PDF processing** (if not already installed):
   ```bash
   pip install opencv-python pillow pytesseract pyzbar pdf2image pandas
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

The crawler will:
- Read URLs from `stMotor.tsv`
- Automatically detect pagination patterns for each site
- Crawl all pages using enhanced pagination
- Extract product information using AI
- Download associated PDFs
- Save results to `output.csv`

### **Input File Format**

The input file should be a TSV (Tab-Separated Values) file with one URL per line:

```
https://omegamotor.com.tr/en/product/?rated_output=1,50;2,20;3,00;4,00;5,50;7,50;10,00;15,00;18,50;22,00;30,00;45,00;55,00&body_size=80;90;110;112;132;160;180;200;225&product_code=&rated_speed=&efficiency_class=IE3&body_material=Al&mountig=&min_rated_voltage=230;400&pol_number=2;4;6;8&service_factor=&view=1&page=0
https://www.bibus.nl/en/cylinders
https://shop.example.com/catalog
```

## 🔧 Enhanced Pagination System

### **Supported Pagination Types**

The crawler automatically detects and handles various pagination patterns:

1. **Page-Based**: `?page=1`, `?p=2`, `?pg=3`
2. **Offset-Based**: `?offset=20`, `?start=40`, `?skip=60`
3. **Limit-Offset**: `?limit=20&offset=40`
4. **Cursor-Based**: `?cursor=abc123`, `?after=xyz789`
5. **Size-Based**: `?size=20&offset=40`
6. **Per-Page**: `?per_page=25&page=2`

### **Automatic Detection**

The system automatically detects pagination patterns by:
- Analyzing URL parameters
- Checking domain-specific patterns (Shopify, WooCommerce, Magento, APIs)
- Examining URL path patterns
- Identifying existing pagination parameters

### **Intelligent Parameter Management**

The enhanced pagination system:
- Preserves all existing URL parameters (filters, sorting, etc.)
- Removes duplicate pagination parameters
- Handles complex query strings with multiple parameters
- Maintains URL integrity across pagination

### **Real-World Examples**

**E-commerce Sites:**
```python
# Shopify Store
url = "https://shop.mystore.com/collections/electronics"
# Automatically detected as page-based pagination
# Crawls: ?page=1, ?page=2, ?page=3, etc.

# WooCommerce Site
url = "https://mywordpress.com/shop/?product_cat=laptops"
# Automatically detected as page-based pagination
# Crawls: ?product_cat=laptops&page=1, ?product_cat=laptops&page=2, etc.
```

**API Endpoints:**
```python
# REST API
url = "https://api.example.com/v1/products?category=electronics"
# Automatically detected as limit-offset pagination
# Crawls: ?category=electronics&limit=20&offset=0, ?category=electronics&limit=20&offset=20, etc.
```

**Complex URLs (Like Omega Motor):**
```python
omega_url = "https://omegamotor.com.tr/en/product/?rated_output=1,50;2,20;3,00;4,00;5,50;7,50;10,00;15,00;18,50;22,00;30,00;45,00;55,00&body_size=80;90;110;112;132;160;180;200;225&product_code=&rated_speed=&efficiency_class=IE3&body_material=Al&mountig=&min_rated_voltage=230;400&pol_number=2;4;6;8&service_factor=&view=1&page=0"

# Automatically detected as page-based pagination
# Preserves all complex parameters while managing pagination
# Next page: same URL with page=1 instead of page=0
```

## 📊 Output

### **CSV Output**
The crawler generates a CSV file with extracted product data:
```csv
productName,productLink
"IE3 3kW 1000rpm, Al 132S6, 3Ph 50Hz 400V/690V D/Y, IMB5",https://omegamotor.com.tr/en/product/detail/561
"IE3 3kW 1000rpm, Al 132S6, 3Ph 50Hz 400V/690V D/Y, IMB14",https://omegamotor.com.tr/en/product/detail/563
"SAI Cylinders with groove profile",https://www.bibus.nl/en/sai-cylinders-with-groove-profile-ma01
```

### **PDF Downloads**
PDFs are organized by product name in the `pdfs/` directory:
```
pdfs/
├── IE3 3kW 1000rpm, Al 132S6, 3Ph 50Hz 400V/690V D/Y, IMB5/
│   ├── catalog.pdf
│   ├── catalog.pdf_cleaned.pdf
│   └── datasheet.pdf
└── SAI Cylinders with groove profile/
    ├── manual.pdf
    └── manual.pdf_cleaned.pdf
```

## ⚙️ Configuration

### **Core Configuration** (`config.py`)
```python
REQUIRED_KEYS = [
    "productName",
    "productLink",
]
```

### **Browser Configuration** (`utils/scraper_utils.py`)
```python
def get_browser_config() -> BrowserConfig:
    return BrowserConfig(
        browser_type="chromium",
        headless=False,  # Set to True for production
        verbose=True,    # Enable debug logging
        headers={"User-Agent": user_agent},
    )
```

### **LLM Configuration**
```python
def get_llm_strategy() -> LLMExtractionStrategy:
    return LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="groq/deepseek-r1-distill-llama-70b",
            api_token="your_api_token_here",
        ),
        schema=Venue.model_json_schema(),
        extraction_type="schema",
        instruction="Extract ALL products from this webpage...",
        input_format="markdown",
        verbose=True,
    )
```

## 🔍 Troubleshooting

### **Common Issues**

1. **No products extracted**:
   - Check if the website uses JavaScript rendering
   - Verify the URL is accessible
   - Check for anti-bot protection
   - Ensure the LLM API key is valid

2. **Pagination not working**:
   - The enhanced pagination system should automatically detect patterns
   - Check if the website uses a custom pagination system
   - Verify the URL format in the input file

3. **PDF downloads failing**:
   - Check file size limits
   - Verify PDF links are accessible
   - Check for authentication requirements
   - Ensure Tesseract is installed for PDF processing

4. **Import errors**:
   - Install all required dependencies: `pip install -r requirements.txt`
   - Install additional PDF processing dependencies if needed
   - Check Python version compatibility

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

### **Environment Setup**

For PDF processing, ensure you have:
- **Tesseract OCR**: Install from https://github.com/tesseract-ocr/tesseract
- **Poppler**: For PDF to image conversion
- **OpenCV**: For image processing
- **Pillow**: For image manipulation

## 📈 Performance

### **Optimizations**
- **Intelligent Delays**: Random delays between requests to be respectful
- **Session Reuse**: Efficient session handling for better performance
- **Duplicate Prevention**: Prevents downloading identical PDFs
- **Graceful Error Handling**: Continues crawling even if individual pages fail

### **Rate Limiting**
- Configurable delays between pages (3-7 seconds)
- Respectful crawling practices
- Automatic retry mechanisms

## 🔧 Advanced Usage

### **Custom Pagination Patterns**

If you need to handle custom pagination patterns, you can modify the `append_page_param` function in `utils/scraper_utils.py`:

```python
def append_page_param(base_url: str, page_number: int, pagination_type: str = "auto") -> str:
    # Add your custom pagination logic here
    pass
```

### **Custom Data Extraction**

Modify the LLM instruction in `get_llm_strategy()` to extract different data:

```python
instruction=(
    "Extract ALL products from this webpage. Focus on product grids, cards, and catalog items.\n\n"
    "EXTRACT:\n"
    "- productName: Complete product name/title\n"
    "- productLink: Full URL to product detail page\n\n"
    "Add your custom extraction requirements here..."
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
- **OpenCV**: For image processing capabilities
- **Tesseract**: For OCR functionality

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the code comments for configuration options
3. Open an issue on the repository
4. Check the example output files for expected formats
