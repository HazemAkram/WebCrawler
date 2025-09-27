# Configuration Guide

## Overview

This guide covers all configuration options available in the DeepSeek AI Web Crawler, from basic settings to advanced customization options.

## Configuration Files

### Environment Variables (.env)

Create a `.env` file in the project root for sensitive configuration:

```env
# API Keys
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Flask Configuration
FLASK_SECRET_KEY=your_secret_key_here
FLASK_APP_PASSWORD=your_app_password_here
FLASK_DEBUG=false

# System Dependencies (Optional)
TESSERACT_PATH=/usr/local/bin/tesseract
POPPLER_PATH=/usr/local/bin

# Processing Settings
MAX_CONCURRENT_REQUESTS=5
DEFAULT_TIMEOUT=30
```

### Configuration File (config.py)

The main configuration file contains default settings:

```python
# config.py

# Required keys for product extraction
REQUIRED_KEYS = [
    "productLink",
]

# PDF processing keys
PDF_KEYS = [
    "url",
    "text", 
    "type",
    "language",
    "priority",
    "productName",
]

# Default configuration settings
DEFAULT_CONFIG = {
    "output_folder": "output",
    "default_model": "groq/llama-3.1-8b-instant",
    "available_models": [
        "groq/llama-3.1-8b-instant"
    ],
    "crawler_settings": {
        "page_timeout": 30000,
        "max_pages": 10,
        "delay_min": 3,
        "delay_max": 15
    },
    "pdf_settings": {
        "max_file_size_mb": 60,
        "skip_large_files": True
    }
}

# Environment variable names
ENV_VARS = {
    "GROQ_API_KEY": "GROQ_API_KEY",
    "OPENAI_API_KEY": "OPENAI_API_KEY", 
    "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY"
}
```

## CSV Configuration

### Basic Structure

Your CSV file must contain these columns:

```csv
url,cat_name,css_selector,pdf_selector,name_selector,button_selector
"https://example.com/products","Category A","div.product-item","a.pdf-link","h1.product-title","button.load-more"
```

### Column Descriptions

#### `url` (Required)
The starting URL for the product catalog.

**Examples**:
```csv
"https://store.com/products"
"https://catalog.example.com/category/valves"
"https://shop.company.com/products?category=pumps"
```

#### `cat_name` (Required)
Category name for organizing output files.

**Examples**:
```csv
"Industrial Valves"
"A1-Pumps-Centrifugal"
"Motors-AC-3Phase"
"Control Systems"
```

#### `css_selector` (Required)
CSS selector(s) for identifying product elements on the page.

**Single Selector**:
```csv
"div.product-item"
```

**Multiple Selectors** (use `|` to separate):
```csv
"div.product-item|li.product-card|article.product"
```

**Complex Selectors**:
```csv
"div[class*='product']:not(.advertisement)|li.product:has(.price)"
```

#### `pdf_selector` (Required)
CSS selector(s) for identifying PDF download links.

**Common Patterns**:
```csv
"a[href*='.pdf']"
"a[href$='.pdf']"
"button[onclick*='pdf']"
".download-pdf|.datasheet-link"
```

#### `name_selector` (Required)
CSS selector for extracting product names.

**Examples**:
```csv
"h1.product-title"
".product-name"
"h2.title, h3.title"
```

#### `button_selector` (Optional)
CSS selector for pagination buttons (JavaScript-based pagination).

**Examples**:
```csv
"button.load-more"
"a.next-page"
"button#show-more"
".pagination .next:not(.disabled)"
```

### Advanced CSV Examples

#### E-commerce Site
```csv
url,cat_name,css_selector,pdf_selector,name_selector,button_selector
"https://industrial-store.com/pumps","Centrifugal Pumps","div.product-card:not(.sponsored)","a[href*='datasheet']:not(.sample)","h2.product-title","button.load-more"
```

#### Catalog Site with Complex Structure
```csv
url,cat_name,css_selector,pdf_selector,name_selector,button_selector
"https://catalog.company.com/products","Industrial Valves","div[class*='product']:not(.ad):not(.banner)","a[href$='.pdf']:not([href*='sample'])","h1.title, h2.title","a.next:not(.disabled)"
```

#### Multi-language Site
```csv
url,cat_name,css_selector,pdf_selector,name_selector,button_selector
"https://international.com/en/products","Motors","div.product-item","a[href*='manual']:not([href*='demo'])","h1.product-name","button[data-action='load-more']"
```

## Browser Configuration

### Default Settings

```python
def get_browser_config() -> BrowserConfig:
    return BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=False,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        viewport={"width": 1080, "height": 720},
        accept_downloads=True,
        downloads_path="downloads",
        extra_args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-web-security"
        ]
    )
```

### Customization Options

#### Headless Mode
```python
# For debugging - set to False to see browser
headless=False
```

#### User Agent
```python
# Custom user agent
user_agent="Custom Bot 1.0"
```

#### Viewport Size
```python
# Custom viewport
viewport={"width": 1920, "height": 1080}
```

#### Additional Arguments
```python
extra_args=[
    "--disable-images",  # Faster loading
    "--disable-javascript",  # For simple sites
    "--disable-plugins",
    "--disable-extensions"
]
```

## LLM Configuration

### Model Selection

#### Groq Models (Recommended)
```python
GROQ_MODELS = [
    "groq/llama-3.1-8b-instant",      # Fast, cost-effective
    "groq/llama-3.1-70b-versatile",   # Higher quality
    "groq/mixtral-8x7b-32768",        # Balanced performance
]
```

#### OpenAI Models
```python
OPENAI_MODELS = [
    "openai/gpt-4o",                  # High quality
    "openai/gpt-4o-mini",             # Cost-effective
    "openai/gpt-3.5-turbo",           # Fast processing
]
```

#### Anthropic Models
```python
ANTHROPIC_MODELS = [
    "anthropic/claude-3-5-sonnet-20241022",  # Best quality
    "anthropic/claude-3-haiku-20240307",     # Fast processing
]
```

### Model Configuration

```python
def get_llm_strategy(api_key: str = None, model: str = "groq/llama-3.1-8b-instant"):
    if model.startswith("groq/"):
        return LLMConfig(
            provider="groq",
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            model=model,
            temperature=0.1,
            max_tokens=4000,
            timeout=30
        )
    # Additional providers...
```

## PDF Processing Configuration

### Size Limits

```python
PDF_SETTINGS = {
    "max_file_size_mb": 60,           # Maximum PDF file size
    "skip_large_files": True,         # Skip files exceeding limit
    "max_pages_per_pdf": 100,         # Maximum pages to process
}
```

### Processing Options

```python
PDF_PROCESSING = {
    "remove_qr_codes": True,          # Remove QR codes
    "remove_contact_info": True,      # Remove contact information
    "remove_footers": True,           # Remove footer areas
    "add_cover_page": True,           # Add custom cover page
    "enhance_images": True,           # Enhance images for processing
}
```

### OCR Configuration

```python
OCR_SETTINGS = {
    "confidence_threshold": 0,        # Minimum OCR confidence (0-100)
    "bottom_region_ratio": 0.10,      # Process bottom 10% for OCR
    "region_start_ratio": 0.90,       # Start OCR at 90% height
    "enhancement_factor": 2,          # Image enhancement scale
}
```

### Background Estimation

```python
BACKGROUND_SETTINGS = {
    "sample_margin": 40,              # Background sampling margin
    "qr_padding": 10,                 # Padding around QR codes
    "text_padding": 1,                # Padding around text
}
```

## Pagination Configuration

### Automatic Detection

The system automatically detects pagination types:

```python
PAGINATION_TYPES = [
    "page",      # ?page=1, ?page=2
    "offset",    # ?offset=20, ?offset=40
    "limit",     # ?limit=20&offset=40
    "cursor",    # ?cursor=abc123
    "javascript" # Dynamic button-based pagination
]
```

### Custom Pagination Patterns

Add custom patterns in `utils/scraper_utils.py`:

```python
def append_page_param(base_url: str, page_number: int, pagination_type: str = "auto") -> str:
    if pagination_type == "custom":
        # Custom pagination logic
        return f"{base_url}?custom_page={page_number}"
    # Existing patterns...
```

## Performance Configuration

### Concurrent Processing

```python
PERFORMANCE_SETTINGS = {
    "max_concurrent_requests": 5,     # Maximum concurrent requests
    "request_timeout": 30,            # Request timeout in seconds
    "retry_attempts": 3,              # Number of retry attempts
    "retry_delay": 5,                 # Delay between retries
}
```

### Rate Limiting

```python
RATE_LIMITING = {
    "min_delay": 3,                   # Minimum delay between requests
    "max_delay": 15,                  # Maximum delay between requests
    "pdf_delay": 10,                  # Delay between PDF downloads
    "respect_robots_txt": True,       # Respect robots.txt
}
```

### Memory Management

```python
MEMORY_SETTINGS = {
    "max_memory_mb": 2048,            # Maximum memory usage
    "cleanup_interval": 10,           # Cleanup every N products
    "gc_threshold": 100,              # Garbage collection threshold
}
```

## Security Configuration

### Authentication

```python
AUTH_CONFIG = {
    "require_password": True,         # Require password for web interface
    "session_timeout": 3600,          # Session timeout in seconds
    "max_login_attempts": 5,          # Maximum login attempts
    "password_min_length": 8,         # Minimum password length
}
```

### File Security

```python
FILE_SECURITY = {
    "allowed_extensions": [".csv"],   # Allowed upload extensions
    "max_file_size": 100 * 1024 * 1024,  # 100MB max file size
    "sanitize_filenames": True,       # Sanitize uploaded filenames
    "scan_uploads": True,             # Scan uploaded files
}
```

### Network Security

```python
NETWORK_SECURITY = {
    "verify_ssl": True,               # Verify SSL certificates
    "timeout": 30,                    # Network timeout
    "max_redirects": 5,               # Maximum redirects
    "user_agent_rotation": True,      # Rotate user agents
}
```

## Logging Configuration

### Log Levels

```python
LOGGING_CONFIG = {
    "level": "INFO",                  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "crawler.log",            # Log file path
    "max_size": 10 * 1024 * 1024,    # 10MB max log file size
    "backup_count": 5,                # Number of backup log files
}
```

### Real-time Logging

```python
REALTIME_LOGGING = {
    "web_interface": True,            # Enable web interface logging
    "max_log_entries": 1000,          # Maximum log entries in memory
    "log_retention": 24,              # Log retention in hours
}
```

## Output Configuration

### File Organization

```python
OUTPUT_CONFIG = {
    "base_folder": "output",          # Base output folder
    "csv_folder": "CSVS",             # CSV reports folder
    "archive_folder": "archives",     # Archive folder
    "temp_folder": "temp",            # Temporary files folder
}
```

### Naming Conventions

```python
NAMING_CONFIG = {
    "sanitize_names": True,           # Sanitize product names for folders
    "max_name_length": 100,           # Maximum name length
    "timestamp_format": "%Y%m%d_%H%M%S",  # Timestamp format
    "csv_prefix": "downloaded_",      # CSV file prefix
}
```

### Archive Settings

```python
ARCHIVE_CONFIG = {
    "auto_archive": True,             # Automatically create archives
    "archive_format": "tar.gz",       # Archive format
    "compression_level": 6,           # Compression level (1-9)
    "retention_days": 30,             # Archive retention period
}
```

## Environment-Specific Configuration

### Development

```python
# .env.development
FLASK_DEBUG=true
LOG_LEVEL=DEBUG
HEADLESS_MODE=false
API_RATE_LIMIT=1000
```

### Production

```python
# .env.production
FLASK_DEBUG=false
LOG_LEVEL=INFO
HEADLESS_MODE=true
API_RATE_LIMIT=100
SECRET_KEY=production_secret_key
```

### Testing

```python
# .env.testing
FLASK_DEBUG=true
LOG_LEVEL=DEBUG
TEST_MODE=true
MOCK_API_CALLS=true
```

## Configuration Validation

### Schema Validation

```python
from pydantic import BaseModel, Field
from typing import Optional

class CrawlerConfig(BaseModel):
    output_folder: str = Field(default="output")
    default_model: str = Field(default="groq/llama-3.1-8b-instant")
    max_pages: int = Field(default=10, ge=1, le=1000)
    delay_min: int = Field(default=3, ge=1, le=60)
    delay_max: int = Field(default=15, ge=1, le=300)
    pdf_size_limit: int = Field(default=60, ge=1, le=500)
```

### Runtime Validation

```python
def validate_config(config: dict) -> bool:
    """Validate configuration at runtime."""
    try:
        CrawlerConfig(**config)
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False
```

## Best Practices

### Configuration Management

1. **Use Environment Variables**: Store sensitive data in environment variables
2. **Validate Input**: Always validate configuration values
3. **Provide Defaults**: Include sensible default values
4. **Document Settings**: Document all configuration options
5. **Version Control**: Keep configuration templates in version control

### Security Considerations

1. **Protect API Keys**: Never commit API keys to version control
2. **Limit Access**: Restrict access to configuration files
3. **Validate Inputs**: Validate all user inputs
4. **Use HTTPS**: Use HTTPS in production environments
5. **Regular Updates**: Keep dependencies and configuration updated

### Performance Optimization

1. **Tune Parameters**: Adjust settings based on your use case
2. **Monitor Resources**: Monitor memory and CPU usage
3. **Batch Processing**: Use appropriate batch sizes
4. **Cache Results**: Cache results when possible
5. **Optimize Selectors**: Use efficient CSS selectors

Remember: Good configuration is the foundation of a reliable and maintainable system. Take time to understand and properly configure all settings for your specific use case.
