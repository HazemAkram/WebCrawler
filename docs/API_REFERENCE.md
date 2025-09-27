# API Reference

## Overview

The DeepSeek AI Web Crawler provides both a RESTful API and programmatic interfaces for integration with other systems. This document covers all available endpoints, methods, and usage examples.

## Base URL

```
http://localhost:5000
```

## Authentication

### Optional Password Protection

If `FLASK_APP_PASSWORD` environment variable is set, the API requires authentication:

```bash
# Login endpoint
POST /login
Content-Type: application/x-www-form-urlencoded

password=your_password
```

### Session Management

After successful login, a session cookie is set for subsequent requests.

## API Endpoints

### Web Interface Endpoints

#### `GET /`
Returns the main web interface HTML page.

**Response**: HTML page with the crawling interface

#### `GET /login`
Returns the login form (if password protection is enabled).

**Response**: HTML login form

#### `POST /login`
Authenticates user with password.

**Request Body**:
```json
{
  "password": "your_password"
}
```

**Response**:
- `200`: Redirect to main page
- `401`: Invalid password

---

### File Management Endpoints

#### `POST /upload`
Upload a CSV configuration file.

**Request**:
- Content-Type: `multipart/form-data`
- Body: CSV file

**Response**:
```json
{
  "success": true,
  "filename": "sites.csv",
  "filepath": "uploads/sites.csv",
  "total_sites": 5
}
```

**Error Responses**:
```json
{
  "error": "No file provided"
}
```

#### `GET /files`
Returns the file explorer interface for browsing output files.

**Response**: HTML file explorer page

#### `GET /files/<path:subpath>`
Browse specific directories in the output structure.

**Parameters**:
- `subpath`: Directory path (e.g., `output/Category A/Product 1`)

**Response**: HTML file explorer for the specified path

#### `DELETE /delete_item`
Delete a file or directory from the output structure.

**Request Body**:
```json
{
  "path": "output/Category A/Product 1"
}
```

**Response**:
```json
{
  "success": true,
  "message": "File 'output/Category A/Product 1' deleted successfully"
}
```

---

### Crawling Control Endpoints

#### `POST /start_crawling`
Start the web crawling process.

**Request Body**:
```json
{
  "csv_filepath": "uploads/sites.csv",
  "api_key": "your_api_key",
  "pdf_size_limit": 60,
  "skip_large_files": true
}
```

**Response**:
```json
{
  "success": true,
  "message": "Crawling started"
}
```

**Error Responses**:
```json
{
  "error": "Crawling is already running"
}
```

#### `POST /stop_crawling`
Stop the currently running crawling process.

**Response**:
```json
{
  "success": true,
  "message": "Stop request sent"
}
```

#### `GET /status`
Get the current status of the crawling process.

**Response**:
```json
{
  "is_running": true,
  "current_site": 2,
  "total_sites": 5,
  "current_page": 3,
  "total_venues": 45,
  "elapsed_time": 125.5,
  "logs": [
    {
      "timestamp": "14:30:25",
      "level": "INFO",
      "message": "Processing site 2 of 5"
    }
  ],
  "stop_requested": false
}
```

#### `GET /logs`
Get all log messages from the current session.

**Response**:
```json
{
  "logs": [
    {
      "timestamp": "14:30:25",
      "level": "INFO",
      "message": "Starting web crawler..."
    },
    {
      "timestamp": "14:30:26",
      "level": "SUCCESS",
      "message": "Crawling completed successfully!"
    }
  ]
}
```

---

### Product Processing Endpoints

#### `GET /products`
Returns the product-only processing interface.

**Response**: HTML page for product-only downloads

#### `POST /upload_products`
Upload a CSV file for product-only processing.

**Request**:
- Content-Type: `multipart/form-data`
- Body: CSV file with product URLs

**Response**:
```json
{
  "success": true,
  "filename": "products.csv",
  "filepath": "uploads/products.csv",
  "total_urls": 25
}
```

#### `POST /start_products`
Start product-only processing (PDF downloads without crawling).

**Request Body**:
```json
{
  "csv_filepath": "uploads/products.csv",
  "api_key": "your_api_key",
  "pdf_size_limit": 60
}
```

**Response**:
```json
{
  "success": true,
  "message": "Product installation started"
}
```

---

### Download and Archive Endpoints

#### `GET /download_output`
Download the complete output folder as a ZIP file.

**Response**: ZIP file download

#### `GET /download_output?mode=link`
Create a persistent archive and return download link.

**Response**:
```json
{
  "success": true,
  "archive_url": "/archives/crawler_output_20250101_120000.tar.gz",
  "archive_path": "/path/to/archive.tar.gz"
}
```

#### `GET /archives/<filename>`
Download a specific archive file.

**Parameters**:
- `filename`: Archive filename (e.g., `crawler_output_20250101_120000.tar.gz`)

**Response**: Archive file download

---

### System Information Endpoints

#### `GET /server-info`
Get server system information and statistics.

**Response**:
```json
{
  "python_version": "3.9.7",
  "platform": "Windows-10-10.0.19041-SP0",
  "cpu_count": 8,
  "cpu_percent": 15.2,
  "memory_total": "16.0 GB",
  "memory_available": "8.5 GB",
  "memory_percent": 46.8,
  "disk_total": "500.0 GB",
  "disk_free": "250.0 GB",
  "disk_percent": 50.0,
  "output_folder_size": "2.5 GB"
}
```

---

## Programmatic API

### Python Integration

#### Main Crawling Function

```python
from main import crawl_from_sites_csv

# Basic usage
await crawl_from_sites_csv("sites.csv")

# With API key
await crawl_from_sites_csv(
    input_file="sites.csv",
    api_key="your_api_key",
    model="groq/llama-3.1-8b-instant"
)

# With callbacks for monitoring
def status_callback(status):
    print(f"Progress: {status['current_site']}/{status['total_sites']}")

def stop_callback():
    return False  # Return True to stop crawling

await crawl_from_sites_csv(
    input_file="sites.csv",
    api_key="your_api_key",
    status_callback=status_callback,
    stop_requested_callback=stop_callback
)
```

#### PDF Processing Function

```python
from cleaner import pdf_processing

# Process a single PDF
pdf_processing(
    file_path="path/to/document.pdf",
    api_key="your_api_key"
)

# With logging callback
def log_callback(message, level):
    print(f"[{level}] {message}")

pdf_processing(
    file_path="path/to/document.pdf",
    api_key="your_api_key",
    log_callback=log_callback
)
```

#### Configuration Management

```python
from config import DEFAULT_CONFIG
from utils.scraper_utils import set_pdf_size_limit

# Update PDF size limit
set_pdf_size_limit(100)  # 100MB limit

# Access configuration
print(DEFAULT_CONFIG['crawler_settings']['max_pages'])
```

---

## Error Handling

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (invalid password)
- `403`: Forbidden (access denied)
- `404`: Not Found (file/resource not found)
- `500`: Internal Server Error

### Error Response Format

```json
{
  "error": "Error description",
  "details": "Additional error information",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

### Common Error Scenarios

#### Invalid CSV Format
```json
{
  "error": "Error reading CSV: Invalid column headers"
}
```

#### Missing API Key
```json
{
  "error": "API key is required"
}
```

#### File Not Found
```json
{
  "error": "Invalid CSV file path"
}
```

#### Processing Errors
```json
{
  "error": "Error during crawling: Network timeout"
}
```

---

## Rate Limiting

### Built-in Delays

The crawler implements respectful delays between requests:
- **Page Navigation**: 3-15 seconds random delay
- **PDF Downloads**: 10-25 seconds between downloads
- **API Calls**: Respects provider rate limits

### Configuration

```python
# Modify delays in config.py
DEFAULT_CONFIG = {
    "crawler_settings": {
        "delay_min": 5,    # Minimum delay in seconds
        "delay_max": 20,   # Maximum delay in seconds
    }
}
```

---

## WebSocket Support

### Real-time Updates

The web interface uses polling for real-time updates. Future versions may include WebSocket support for instant notifications.

### Current Implementation

```javascript
// Polling example
setInterval(async () => {
    const response = await fetch('/status');
    const status = await response.json();
    updateUI(status);
}, 2000); // Poll every 2 seconds
```

---

## SDK and Client Libraries

### Python Client

```python
import requests

class DeepSeekCrawler:
    def __init__(self, base_url="http://localhost:5000", password=None):
        self.base_url = base_url
        self.session = requests.Session()
        
        if password:
            self.login(password)
    
    def login(self, password):
        response = self.session.post(
            f"{self.base_url}/login",
            data={"password": password}
        )
        return response.status_code == 200
    
    def upload_csv(self, file_path):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(
                f"{self.base_url}/upload",
                files=files
            )
        return response.json()
    
    def start_crawling(self, csv_filepath, api_key, **kwargs):
        data = {
            "csv_filepath": csv_filepath,
            "api_key": api_key,
            **kwargs
        }
        response = self.session.post(
            f"{self.base_url}/start_crawling",
            json=data
        )
        return response.json()
    
    def get_status(self):
        response = self.session.get(f"{self.base_url}/status")
        return response.json()

# Usage
crawler = DeepSeekCrawler(password="your_password")
result = crawler.upload_csv("sites.csv")
status = crawler.start_crawling(
    csv_filepath=result["filepath"],
    api_key="your_api_key"
)
```

---

## Testing

### API Testing with curl

```bash
# Upload CSV file
curl -X POST -F "file=@sites.csv" http://localhost:5000/upload

# Start crawling
curl -X POST -H "Content-Type: application/json" \
  -d '{"csv_filepath":"uploads/sites.csv","api_key":"your_key"}' \
  http://localhost:5000/start_crawling

# Check status
curl http://localhost:5000/status

# Download output
curl -O http://localhost:5000/download_output
```

### Unit Testing

```python
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_upload_csv(client):
    with open('test_sites.csv', 'rb') as f:
        response = client.post('/upload', data={'file': f})
    assert response.status_code == 200
    assert 'success' in response.json

def test_status_endpoint(client):
    response = client.get('/status')
    assert response.status_code == 200
    assert 'is_running' in response.json
```

---

## Best Practices

### API Usage

1. **Always check response status codes**
2. **Handle errors gracefully with retry logic**
3. **Use appropriate timeouts for long-running operations**
4. **Monitor progress using status endpoints**
5. **Clean up temporary files after processing**

### Security

1. **Use HTTPS in production**
2. **Implement proper authentication**
3. **Validate all input parameters**
4. **Sanitize file paths and names**
5. **Limit file upload sizes**

### Performance

1. **Use appropriate batch sizes**
2. **Implement caching where possible**
3. **Monitor resource usage**
4. **Use async operations for I/O**
5. **Implement proper error recovery**
