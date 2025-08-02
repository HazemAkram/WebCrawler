# DeepSeek AI Web Crawler

A powerful, AI-powered web crawler that extracts product information from e-commerce websites and downloads associated PDF documents. Features intelligent pagination handling, duplicate detection, and advanced PDF processing with QR code removal and text cleaning.

## 🚀 Features

### **AI-Powered Product Extraction**
- **LLM-Based Extraction**: Uses Groq/DeepSeek R1 Distill Llama 70B for intelligent data extraction
- **Universal Compatibility**: Works with any website layout without custom configuration
- **Schema Validation**: Ensures extracted data meets required format using Pydantic models
- **Duplicate Prevention**: Prevents processing duplicate products using name-based detection

### **Advanced Pagination System**
- **Multiple Pagination Types**: Supports page-based, offset-based, limit-offset, and cursor-based pagination
- **Automatic Detection**: Detects pagination patterns from URL parameters
- **JavaScript Pagination**: Handles dynamic pagination using button clicks and JavaScript execution
- **Intelligent Parameter Management**: Preserves existing URL parameters while managing pagination

### **Comprehensive PDF Processing**
- **Automatic Download**: Downloads PDFs from product detail pages
- **Duplicate Prevention**: Prevents downloading identical files using content hashing
- **QR Code Removal**: Automatically detects and removes QR codes from PDFs
- **Text Cleaning**: Removes specific text with background color matching
- **Cover Page Addition**: Automatically adds custom cover pages to processed PDFs
- **Smart Organization**: Organizes PDFs by product name in dedicated folders

### **Robust Error Handling**
- **Graceful Degradation**: Continues crawling if individual pages fail
- **Rate Limiting**: Respectful crawling with configurable delays (3-15 seconds)
- **Session Management**: Efficient session handling for better performance
- **SSL Handling**: Configurable SSL verification for various server configurations

## 📁 Project Structure

```
deepseek-ai-web-crawler/
├── main.py                    # Main entry point for the crawler
├── config.py                  # Configuration constants (required keys)
├── cleaner.py                 # PDF processing and cleaning functionality
├── sites.csv                  # Input file with URLs and selectors
├── cover.png                  # Cover page template for PDFs
├── models/
│   └── venue.py              # Defines the Venue data model using Pydantic
├── utils/
│   ├── __init__.py           # Package marker for utils
│   ├── data_utils.py         # Utility functions for data validation
│   └── scraper_utils.py      # Enhanced pagination and crawling utilities
├── pdfs/                     # Output directory for downloaded PDFs
├── requirements.txt          # Python package dependencies
└── README.MD                # This file
```

## 🛠️ Installation

### **Prerequisites**
- Python 3.8 or higher
- Tesseract OCR (for PDF processing)
- Poppler (for PDF to image conversion)

### **Setup Steps**

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd deepseek-ai-web-crawler
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies**:

   **Windows:**
   ```bash
   # Install Tesseract OCR
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   
   # Install Poppler
   # Download from: https://github.com/oschwartz10612/poppler-windows/releases
   ```

   **macOS:**
   ```bash
   brew install tesseract poppler
   ```

   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt-get install tesseract-ocr poppler-utils
   ```

4. **Set up environment variables** (optional):
   Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
   
   **Note**: You can also enter your API key directly in the GUI application.

5. **Test dependencies** (recommended):
   Run the dependency test script to verify everything is working:
   ```bash
   python test_dependencies.py
   ```
   
   This will automatically detect Tesseract and Poppler installations and verify they're working correctly.

## 🚀 Usage

### **Basic Usage**

1. **Prepare your input file** (`sites.csv`):
   ```csv
   url,css_selector,button_selector
   "https://example-store.com/products","div.product-item","button.load-more"
   "https://another-shop.com/catalog","li.product","a.next-page"
   ```

2. **Run the crawler**:

   **Option A: GUI Interface (Recommended)**
   ```bash
   python web_crawler_gui.py
   ```
   
   **Option B: Command Line**
   ```bash
   python main.py
   ```

The crawler will:
- Read URLs and selectors from `sites.csv`
- Automatically detect pagination patterns for each site
- Crawl all pages using enhanced pagination or JavaScript execution
- Extract product information using AI
- Download associated PDFs
- Process and clean PDFs (remove QR codes, add cover pages)
- Save results to CSV files

### **GUI Features**

The GUI application provides:
- **API Configuration**: Enter your API key and select your preferred LLM model
- **File Selection**: Browse and select your CSV file with site configurations
- **Real-time Progress**: Monitor crawling progress with detailed logs
- **API Testing**: Test your API connection before starting the crawl
- **User Preferences**: Automatically saves and loads your settings
- **Environment Integration**: Automatically detects API keys from environment variables

### **Input File Format**

The `sites.csv` file should contain:
- **url**: The starting URL for the product catalog
- **css_selector**: CSS selector(s) for product elements (comma-separated if multiple)
- **button_selector**: CSS selector for pagination button (optional, for JS-based pagination)

**Example:**
```csv
url,css_selector,button_selector
"https://www.vag-group.com/en/products/product-groups/butterfly-valves/butterflyvalves","div.Products__Item",
"https://omegamotor.com.tr/en/product/","div.product-card","button.load-more"
```

## 🔧 Configuration

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
        headless=True,  # Set to False for debugging
        viewport_width=1080,
        viewport_height=720,
        verbose=True,
        user_agent=user_agent,
    )
```

### **LLM Configuration**
The crawler now supports multiple LLM providers and models. You can configure this through the GUI or by modifying the code:

```python
def get_llm_strategy(api_key: str = None, model: str = "groq/deepseek-r1-distill-llama-70b") -> LLMExtractionStrategy:
    return LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider=model,  # User-selected model
            api_token=api_key,  # User-provided API key
        ),
        schema=Venue.model_json_schema(),
        extraction_type="schema",
        instruction="Extract ALL products from this webpage...",
        input_format="markdown",
        verbose=False,
    )
```

**Supported Models:**
- **Groq**: `groq/deepseek-r1-distill-llama-70b`, `groq/llama3-8b-8192`, `groq/llama3-70b-8192`, `groq/mixtral-8x7b-32768`
- **OpenAI**: `openai/gpt-4o`, `openai/gpt-4o-mini`
- **Anthropic**: `anthropic/claude-3-5-sonnet-20241022`, `anthropic/claude-3-haiku-20240307`

## 📊 Output

### **CSV Output**
The crawler generates CSV files with extracted product data:
```csv
productName,productLink
"IE3 3kW 1000rpm, Al 132S6, 3Ph 50Hz 400V/690V D/Y, IMB5",https://omegamotor.com.tr/en/product/detail/561
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

## 🔍 Pagination System

### **Supported Pagination Types**

1. **Page-Based**: `?page=1`, `?p=2`, `?pg=3`
2. **Offset-Based**: `?offset=20`, `?start=40`, `?skip=60`
3. **Limit-Offset**: `?limit=20&offset=40`
4. **Cursor-Based**: `?cursor=abc123`, `?after=xyz789`
5. **JavaScript-Based**: Dynamic pagination using button clicks

### **Automatic Detection**

The system automatically detects pagination patterns by:
- Analyzing URL parameters
- Checking for existing pagination parameters
- Identifying domain-specific patterns
- Supporting JavaScript-based pagination for dynamic sites

### **JavaScript Pagination**

For sites with dynamic pagination, the crawler can:
- Execute JavaScript to click pagination buttons
- Extract content from multiple pages
- Handle infinite scroll or "Load More" functionality
- Process all pages in a single session

## 🧹 PDF Processing Features

### **QR Code Removal**
- Automatically detects QR codes using OpenCV and pyzbar
- Estimates background color for seamless removal
- Handles multiple QR codes per page
- Supports enhanced detection for small QR codes

### **Text Cleaning**
- Removes specific text patterns with background color matching
- Uses OCR to identify text locations
- Preserves document layout and formatting
- Configurable padding around removed text

### **Cover Page Addition**
- Automatically adds custom cover pages to processed PDFs
- Uses `cover.png` as the template
- Maintains PDF quality and structure

## 🔧 Advanced Usage

### **Custom Pagination Patterns**

Modify the `append_page_param` function in `utils/scraper_utils.py`:

```python
def append_page_param(base_url: str, page_number: int, pagination_type: str = "auto") -> str:
    # Add your custom pagination logic here
    pass
```

### **Custom Data Extraction**

Modify the LLM instruction in `get_llm_strategy()`:

```python
instruction=(
    "Extract ALL products from this webpage. Focus on product grids, cards, and catalog items.\n\n"
    "EXTRACT:\n"
    "- productName: Complete product name/title\n"
    "- productLink: Full URL to product detail page\n\n"
    "Add your custom extraction requirements here..."
)
```

### **PDF Processing Customization**

The PDF processing now automatically detects Tesseract and Poppler installations across different operating systems. No manual path configuration is required.

If you need to customize the path detection, modify the `find_tesseract_path()` and `find_poppler_path()` functions in `cleaner.py`:

```python
def find_tesseract_path():
    # Add your custom paths here
    custom_paths = [
        "/path/to/your/tesseract",
        # Add more paths as needed
    ]
    # ... rest of the function
```

### **Automatic Path Detection**

The system automatically detects:
- **Tesseract OCR**: Common installation paths on Windows, macOS, and Linux
- **Poppler**: Common installation paths and PATH environment variable
- **Fallback mechanisms**: System commands (`where` on Windows, `which` on Unix-like systems)

Supported paths:
- **Windows**: Program Files, Program Files (x86), User AppData
- **macOS**: Homebrew, system directories
- **Linux**: System directories, custom installations

## 🔍 Troubleshooting

### **Common Issues**

1. **No products extracted**:
   - Check if the website uses JavaScript rendering
   - Verify the URL is accessible
   - Check for anti-bot protection
   - Ensure the LLM API key is valid
   - Verify CSS selectors are correct

2. **Pagination not working**:
   - Check if the website uses a custom pagination system
   - Verify the URL format in the input file
   - Try JavaScript-based pagination for dynamic sites
   - Check browser console for errors

3. **PDF downloads failing**:
   - Check file size limits
   - Verify PDF links are accessible
   - Check for authentication requirements
   - Ensure Tesseract is installed and configured

4. **Import errors**:
   - Install all required dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)
   - Verify system dependencies (Tesseract, Poppler)

### **Debug Mode**

Enable verbose logging by setting `verbose=True` in the browser configuration:

```python
def get_browser_config() -> BrowserConfig:
    return BrowserConfig(
        browser_type="chromium",
        headless=False,  # Set to False for debugging
        verbose=True,    # Enable debug logging
    )
```

### **Performance Optimization**

- **Rate Limiting**: Adjust delays in `main.py` (currently 3-15 seconds)
- **Session Reuse**: Efficient session handling for better performance
- **Duplicate Prevention**: Prevents downloading identical PDFs
- **Graceful Error Handling**: Continues crawling even if individual pages fail

## 📈 Performance Considerations

### **Optimizations**
- **Intelligent Delays**: Random delays between requests to be respectful
- **Session Reuse**: Efficient session handling for better performance
- **Duplicate Prevention**: Prevents downloading identical PDFs
- **Graceful Error Handling**: Continues crawling even if individual pages fail

### **Rate Limiting**
- Configurable delays between pages (3-15 seconds)
- Respectful crawling practices
- Automatic retry mechanisms

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
- **Poppler**: For PDF processing utilities

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the code comments for configuration options
3. Open an issue on the repository
4. Check the example output files for expected formats

## 🔄 Recent Updates

- Enhanced pagination system with automatic detection
- JavaScript-based pagination for dynamic sites
- Advanced PDF processing with QR code removal
- Improved duplicate detection and prevention
- Better error handling and logging
- Support for complex URL parameters
