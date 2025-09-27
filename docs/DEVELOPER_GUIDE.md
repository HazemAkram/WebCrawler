# Developer Guide

## Overview

This guide is for developers who want to extend, modify, or contribute to the DeepSeek AI Web Crawler. It covers the codebase structure, development setup, and best practices for contributing.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)
- IDE with Python support (VS Code, PyCharm, etc.)

### Initial Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd deepseek-ai-web-crawler
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. **Install development tools**
   ```bash
   pip install pytest black flake8 mypy pre-commit
   ```

### Environment Configuration

Create a `.env` file for development:

```env
# Development settings
FLASK_DEBUG=true
FLASK_SECRET_KEY=dev-secret-key
GROQ_API_KEY=your_dev_api_key
OPENAI_API_KEY=your_dev_openai_key

# Optional: Set custom paths
TESSERACT_PATH=/usr/local/bin/tesseract
POPPLER_PATH=/usr/local/bin
```

## Codebase Structure

### Core Modules

```
deepseek-ai-web-crawler/
├── main.py                 # Main crawling orchestration
├── app.py                  # Flask web application
├── cleaner.py              # PDF processing and cleaning
├── config.py               # Configuration constants
├── download_pdf_links.py   # Product-only PDF processing
├── models/
│   ├── __init__.py
│   └── venue.py           # Pydantic data models
├── utils/
│   ├── __init__.py
│   ├── data_utils.py      # Data validation utilities
│   └── scraper_utils.py   # Web scraping utilities
├── templates/             # HTML templates
├── static/               # CSS/JS assets
├── uploads/              # Upload directory
├── output/               # Output directory
├── CSVS/                 # Generated CSV reports
├── archives/             # Archived results
└── docs/                 # Documentation
```

### Key Components

#### `main.py` - Core Crawling Logic
- **Purpose**: Main orchestration of the crawling process
- **Key Functions**:
  - `crawl_from_sites_csv()`: Main crawling function
  - `read_sites_from_csv()`: CSV parsing and validation
  - `cleanup_crawler_session()`: Resource management

#### `app.py` - Web Interface
- **Purpose**: Flask web application and API endpoints
- **Key Features**:
  - File upload handling
  - Real-time progress monitoring
  - Authentication system
  - File management interface

#### `cleaner.py` - PDF Processing
- **Purpose**: AI-powered PDF content cleaning
- **Key Functions**:
  - `pdf_processing()`: Main processing pipeline
  - `remove_qr_codes_from_pdf()`: QR code detection and removal
  - `replace_text_in_scanned_pdf_ai()`: AI-powered text removal

#### `utils/scraper_utils.py` - Web Scraping Utilities
- **Purpose**: Core scraping functionality
- **Key Functions**:
  - `fetch_and_process_page()`: Page processing with LLM
  - `download_pdf_links()`: PDF discovery and download
  - `append_page_param()`: Pagination URL construction

## Development Workflow

### Code Style

Follow these coding standards:

1. **PEP 8 Compliance**: Use Black for formatting
2. **Type Hints**: Add type annotations for functions
3. **Docstrings**: Document all public functions and classes
4. **Error Handling**: Use specific exceptions and proper logging

### Example Code Structure

```python
"""
Module docstring describing the purpose.
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def example_function(param1: str, param2: Optional[int] = None) -> Dict[str, str]:
    """
    Example function with proper documentation.
    
    Args:
        param1: Description of parameter
        param2: Optional parameter description
        
    Returns:
        Dictionary with results
        
    Raises:
        ValueError: When parameter is invalid
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    
    try:
        # Function implementation
        result = {"status": "success", "data": param1}
        logger.info(f"Function completed successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Function failed: {str(e)}")
        raise
```

### Testing

#### Unit Tests

Create tests in the `tests/` directory:

```python
# tests/test_scraper_utils.py
import pytest
from utils.scraper_utils import sanitize_folder_name


def test_sanitize_folder_name():
    """Test folder name sanitization."""
    # Test basic sanitization
    result = sanitize_folder_name("Test Product/Name")
    assert result == "Test Product_Name"
    
    # Test empty string
    result = sanitize_folder_name("")
    assert result == "Unnamed"
    
    # Test special characters
    result = sanitize_folder_name("Product<>|?*")
    assert "/" not in result
    assert "<" not in result
```

#### Integration Tests

```python
# tests/test_integration.py
import pytest
from main import crawl_from_sites_csv


@pytest.mark.asyncio
async def test_crawl_integration():
    """Test end-to-end crawling process."""
    # Create test CSV
    test_csv = "tests/test_sites.csv"
    
    # Run crawling
    await crawl_from_sites_csv(test_csv)
    
    # Verify results
    assert os.path.exists("output")
    # Add more assertions
```

#### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_scraper_utils.py

# Run with verbose output
pytest -v
```

### Debugging

#### Debug Mode

Enable debug mode in development:

```python
# In app.py
DEBUG_MODE = True
app.config['DEBUG'] = True

# In main.py
def get_browser_config() -> BrowserConfig:
    return BrowserConfig(
        browser_type="chromium",
        headless=False,  # Set to False for debugging
        verbose=True,
    )
```

#### Logging Configuration

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
```

#### Common Debugging Techniques

1. **Browser Debugging**: Set `headless=False` to see browser actions
2. **Log Analysis**: Check log files for error patterns
3. **Step-by-step Execution**: Use debugger to trace execution
4. **Network Monitoring**: Check API calls and responses

## Extending the System

### Adding New LLM Providers

1. **Update Configuration**
   ```python
   # In config.py
   ENV_VARS = {
       "GROQ_API_KEY": "GROQ_API_KEY",
       "OPENAI_API_KEY": "OPENAI_API_KEY",
       "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
       "COHERE_API_KEY": "COHERE_API_KEY",  # New provider
   }
   ```

2. **Add Provider Support**
   ```python
   # In utils/scraper_utils.py
   def get_llm_strategy(api_key: str = None, model: str = "groq/llama-3.1-8b-instant"):
       if model.startswith("cohere/"):
           return LLMConfig(
               provider="cohere",
               api_key=api_key or os.getenv("COHERE_API_KEY"),
               model=model,
               # Provider-specific configuration
           )
       # Existing providers...
   ```

### Adding New PDF Processing Steps

1. **Create Processing Function**
   ```python
   # In cleaner.py
   def custom_pdf_processing(images: List[Image.Image], config: Dict) -> List[Image.Image]:
       """
       Custom PDF processing step.
       
       Args:
           images: List of PIL Images
           config: Processing configuration
           
       Returns:
           List of processed PIL Images
       """
       processed_images = []
       for img in images:
           # Custom processing logic
           processed_img = custom_processing_step(img, config)
           processed_images.append(processed_img)
       return processed_images
   ```

2. **Integrate into Pipeline**
   ```python
   def pdf_processing(file_path: str, api_key: str, log_callback=None):
       # Existing processing steps...
       
       # Add custom processing
       final_images = custom_pdf_processing(text_removed, custom_config)
       
       # Continue with existing pipeline...
   ```

### Adding New Output Formats

1. **Create Output Handler**
   ```python
   # In utils/output_utils.py
   def export_to_json(products: List[Dict], output_path: str):
       """Export products to JSON format."""
       import json
       
       data = {
           "timestamp": datetime.now().isoformat(),
           "total_products": len(products),
           "products": products
       }
       
       with open(output_path, 'w') as f:
           json.dump(data, f, indent=2)
   ```

2. **Integrate with Main Process**
   ```python
   # In main.py
   async def crawl_from_sites_csv(...):
       # Existing crawling logic...
       
       # Export to additional formats
       export_to_json(all_venues, f"output/products_{timestamp}.json")
   ```

### Adding New Interface Types

#### CLI Interface

```python
# cli.py
import argparse
import asyncio
from main import crawl_from_sites_csv


def main():
    parser = argparse.ArgumentParser(description="DeepSeek AI Web Crawler")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--model", default="groq/llama-3.1-8b-instant", help="LLM model")
    parser.add_argument("--output", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    asyncio.run(crawl_from_sites_csv(
        input_file=args.input,
        api_key=args.api_key,
        model=args.model
    ))


if __name__ == "__main__":
    main()
```

#### API Wrapper

```python
# api_wrapper.py
from typing import Dict, List
import asyncio
from main import crawl_from_sites_csv


class CrawlerAPI:
    """High-level API wrapper for the crawler."""
    
    def __init__(self, api_key: str, model: str = "groq/llama-3.1-8b-instant"):
        self.api_key = api_key
        self.model = model
    
    async def crawl_sites(self, sites: List[Dict]) -> Dict:
        """Crawl multiple sites programmatically."""
        # Convert sites to CSV format
        csv_content = self._sites_to_csv(sites)
        
        # Write temporary CSV
        temp_csv = "temp_sites.csv"
        with open(temp_csv, 'w') as f:
            f.write(csv_content)
        
        try:
            # Run crawling
            await crawl_from_sites_csv(
                input_file=temp_csv,
                api_key=self.api_key,
                model=self.model
            )
            
            # Return results summary
            return self._get_results_summary()
        finally:
            # Clean up temporary file
            os.remove(temp_csv)
```

## Performance Optimization

### Profiling

Use Python profiling tools to identify bottlenecks:

```python
import cProfile
import pstats

def profile_crawling():
    """Profile the crawling process."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run crawling code
    asyncio.run(crawl_from_sites_csv("test.csv"))
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
```

### Memory Optimization

```python
import gc
import psutil

def monitor_memory():
    """Monitor memory usage during processing."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Processing code here
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory usage: {final_memory - initial_memory:.2f} MB")
    
    # Force garbage collection
    gc.collect()
```

### Async Optimization

```python
import asyncio
from asyncio import Semaphore

async def bounded_crawl(urls: List[str], max_concurrent: int = 5):
    """Crawl URLs with concurrency limit."""
    semaphore = Semaphore(max_concurrent)
    
    async def crawl_with_semaphore(url: str):
        async with semaphore:
            # Crawling logic here
            return await crawl_single_url(url)
    
    tasks = [crawl_with_semaphore(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

## Security Considerations

### Input Validation

```python
import re
from pathlib import Path

def validate_csv_path(file_path: str) -> bool:
    """Validate CSV file path for security."""
    path = Path(file_path)
    
    # Check for path traversal
    if ".." in str(path):
        return False
    
    # Check file extension
    if not path.suffix.lower() == '.csv':
        return False
    
    # Check if file exists and is readable
    if not path.exists() or not path.is_file():
        return False
    
    return True

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename
```

### API Key Security

```python
import os
from cryptography.fernet import Fernet

class SecureAPIKeyManager:
    """Secure API key management."""
    
    def __init__(self):
        self.key = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
        self.cipher = Fernet(self.key)
    
    def encrypt_key(self, api_key: str) -> str:
        """Encrypt API key for storage."""
        return self.cipher.encrypt(api_key.encode()).decode()
    
    def decrypt_key(self, encrypted_key: str) -> str:
        """Decrypt stored API key."""
        return self.cipher.decrypt(encrypted_key.encode()).decode()
```

## Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads output CSVS archives

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  crawler:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./uploads:/app/uploads
      - ./output:/app/output
      - ./CSVS:/app/CSVS
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - FLASK_SECRET_KEY=${FLASK_SECRET_KEY}
    restart: unless-stopped
```

### Production Configuration

```python
# production_config.py
import os

class ProductionConfig:
    """Production configuration settings."""
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY')
    WTF_CSRF_ENABLED = True
    
    # Performance
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = '/var/log/crawler.log'
    
    # Database (if needed)
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    # External services
    REDIS_URL = os.environ.get('REDIS_URL')
```

## Contributing

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with proper tests and documentation
4. **Run tests**: `pytest`
5. **Check code style**: `black . && flake8`
6. **Commit changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Code Review Guidelines

- **Functionality**: Ensure code works as intended
- **Testing**: Include appropriate tests
- **Documentation**: Update documentation as needed
- **Performance**: Consider performance implications
- **Security**: Review security implications
- **Style**: Follow project coding standards

### Issue Reporting

When reporting issues, include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces
- Sample input data (anonymized)

### Feature Requests

For new features, provide:
- Use case description
- Proposed implementation approach
- Potential impact on existing functionality
- Example usage

## Resources

### Documentation
- [Crawl4AI Documentation](https://github.com/unclecode/crawl4ai)
- [Playwright Documentation](https://playwright.dev/python/)
- [Flask Documentation](https://flask.palletsprojects.com/)

### Tools
- [Black Code Formatter](https://black.readthedocs.io/)
- [Pytest Testing Framework](https://pytest.org/)
- [MyPy Type Checking](https://mypy.readthedocs.io/)

### Best Practices
- [Python Best Practices](https://realpython.com/python-best-practices/)
- [Async Python Patterns](https://realpython.com/async-io-python/)
- [API Design Guidelines](https://restfulapi.net/)

Remember: Good code is not just functional, but also maintainable, testable, and well-documented. Take time to write clean, readable code that others can understand and extend.
