# User Guide

## Getting Started

Welcome to the DeepSeek AI Web Crawler! This guide will help you get up and running quickly, from installation to processing your first batch of products.

## Quick Start

### 1. Installation

Follow the installation steps in the main README, then start the web interface:

```bash
python app.py
```

Open your browser to `http://localhost:5000` to access the dashboard.

### 2. Prepare Your Configuration

Create a CSV file with your target websites. Here's a simple example:

```csv
url,cat_name,css_selector,pdf_selector,name_selector,button_selector
"https://example.com/products","Valves","div.product-item","a.pdf-link","h1.product-title","button.load-more"
```

### 3. Upload and Start

1. Upload your CSV file using the web interface
2. Enter your API key (Groq recommended for best performance)
3. Click "Start Crawling"
4. Monitor progress in real-time

## Detailed Configuration

### CSV File Format

Your CSV file must contain these columns:

| Column | Required | Description | Example |
|--------|----------|-------------|---------|
| `url` | ✅ | Starting URL for product catalog | `https://store.com/products` |
| `cat_name` | ✅ | Category name for organization | `Industrial Valves` |
| `css_selector` | ✅ | CSS selector for product elements | `div.product-item\|li.product-card` |
| `pdf_selector` | ✅ | CSS selector for PDF links | `a[href*=".pdf"]\|button.download` |
| `name_selector` | ✅ | CSS selector for product names | `h1.product-title` |
| `button_selector` | ❌ | CSS selector for pagination buttons | `button.load-more` |

### Advanced CSV Examples

#### Multiple Selectors
Use pipe (`|`) to separate multiple selectors:

```csv
url,cat_name,css_selector,pdf_selector,name_selector,button_selector
"https://store.com/products","Valves","div.product-item|li.product-card","a.pdf-link|button.download","h1.product-title|.product-name","button.load-more|a.next-page"
```

#### JavaScript Pagination
For sites that use JavaScript to load more products:

```csv
url,cat_name,css_selector,pdf_selector,name_selector,button_selector
"https://dynamic-store.com/catalog","Pumps","div.product-item","a.datasheet-link","h2.product-title","button#load-more"
```

#### Complex Selectors
For sites with complex HTML structures:

```csv
url,cat_name,css_selector,pdf_selector,name_selector,button_selector
"https://complex-site.com/products","Motors","div[class*='product']:not(.advertisement)","a[href$='.pdf']:not(.sample)","h1.title, h2.title",".pagination .next:not(.disabled)"
```

## Web Interface Guide

### Dashboard Overview

The web interface provides a comprehensive dashboard with the following sections:

1. **File Upload**: Upload your CSV configuration
2. **Settings**: Configure API keys and processing options
3. **Progress Monitor**: Real-time crawling progress
4. **Log Viewer**: Detailed processing logs
5. **File Explorer**: Browse and download results

### Step-by-Step Process

#### Step 1: Upload Configuration
1. Click "Choose File" and select your CSV file
2. Wait for the upload to complete
3. Review the site count and configuration summary

#### Step 2: Configure Settings
1. Enter your API key in the "API Key" field
2. Select your preferred model (Groq recommended)
3. Adjust PDF size limits if needed
4. Enable/disable large file skipping

#### Step 3: Start Processing
1. Click "Start Crawling" to begin
2. Monitor progress in the status panel
3. Watch real-time logs for detailed information
4. Use "Stop" button if you need to halt processing

#### Step 4: Review Results
1. Check the file explorer for downloaded content
2. Download individual files or complete archives
3. Review CSV reports for processing summaries

### Progress Monitoring

The progress panel shows:
- **Current Site**: Which website is being processed
- **Total Sites**: Total number of sites in your CSV
- **Current Page**: Current page being processed
- **Total Products**: Number of products found so far
- **Elapsed Time**: How long the process has been running

### Log Levels

Logs are color-coded by importance:
- 🔵 **INFO**: General information about the process
- 🟡 **WARNING**: Non-critical issues that don't stop processing
- 🔴 **ERROR**: Critical errors that may affect results
- 🟢 **SUCCESS**: Successful operations and completions

## Desktop GUI Guide

### Launching the Desktop Application

```bash
python web_crawler_gui.py
```

### GUI Features

1. **File Selection**: Browse and select your CSV file
2. **API Configuration**: Enter your API key and select model
3. **Settings Panel**: Configure processing options
4. **Progress Display**: Visual progress bars and status
5. **Log Window**: Real-time log display
6. **Results Browser**: Built-in file browser for results

### GUI Workflow

1. **Select CSV File**: Use the file browser to select your configuration
2. **Enter API Key**: Paste your API key in the designated field
3. **Test Connection**: Click "Test API" to verify your credentials
4. **Configure Options**: Adjust settings as needed
5. **Start Processing**: Click "Start Crawling"
6. **Monitor Progress**: Watch the progress bars and log output
7. **Browse Results**: Use the results browser to view downloaded files

## Command Line Usage

### Basic Command

```bash
python main.py
```

### Advanced Options

```bash
# With custom CSV file
python main.py --input custom_sites.csv

# With specific model
python main.py --model groq/llama-3.1-8b-instant

# With API key
GROQ_API_KEY=your_key python main.py
```

### Command Line Output

The command line interface provides:
- Real-time progress updates
- Detailed logging information
- Error messages and stack traces
- Summary statistics upon completion

## API Key Setup

### Groq (Recommended)

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to your configuration

**Advantages**: Fast, cost-effective, generous free tier

### OpenAI

1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account or sign in
3. Go to API Keys section
4. Create a new secret key
5. Add billing information if needed

**Advantages**: High quality, reliable, extensive model selection

### Anthropic

1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Sign up for an account
3. Navigate to API Keys
4. Create a new API key
5. Add billing information

**Advantages**: Balanced performance, good for complex tasks

## Troubleshooting Common Issues

### No Products Found

**Symptoms**: Crawler completes but finds no products

**Solutions**:
1. Check CSS selectors in your CSV
2. Verify the website is accessible
3. Test with a simple selector like `div` or `li`
4. Check if the site requires JavaScript
5. Verify the URL is correct

**Debug Steps**:
```bash
# Test with a simple selector
"https://example.com/products","Test","div","a","h1",""

# Check browser console for errors
# Look for anti-bot protection
# Verify the page loads correctly
```

### Pagination Not Working

**Symptoms**: Only first page is processed

**Solutions**:
1. Check the `button_selector` in your CSV
2. Verify the pagination mechanism (URL-based vs JavaScript)
3. Test with different pagination types
4. Check for custom pagination patterns

**Debug Steps**:
```bash
# Try JavaScript pagination
"https://example.com/products","Test","div.product","a.pdf","h1","button.load-more"

# Try URL-based pagination
"https://example.com/products?page=1","Test","div.product","a.pdf","h1",""
```

### PDF Downloads Failing

**Symptoms**: Products found but no PDFs downloaded

**Solutions**:
1. Check `pdf_selector` in your CSV
2. Verify PDF links are accessible
3. Check file size limits
4. Ensure PDFs aren't behind authentication

**Debug Steps**:
```bash
# Test with generic PDF selector
"https://example.com/products","Test","div.product","a[href*='.pdf']","h1",""

# Check file size limits in settings
# Verify direct PDF access
# Test with different selector patterns
```

### API Errors

**Symptoms**: Authentication or quota errors

**Solutions**:
1. Verify API key is correct
2. Check API quota and billing
3. Try a different API provider
4. Check rate limiting

**Debug Steps**:
```bash
# Test API key directly
curl -H "Authorization: Bearer your_key" https://api.groq.com/v1/models

# Check API usage in provider console
# Try with a different model
# Verify network connectivity
```

### Performance Issues

**Symptoms**: Slow processing or timeouts

**Solutions**:
1. Reduce concurrent requests
2. Increase timeout settings
3. Check system resources
4. Optimize CSS selectors

**Debug Steps**:
```bash
# Monitor system resources
# Check network connectivity
# Reduce batch sizes
# Use faster API providers
```

## Best Practices

### CSV Configuration

1. **Test Selectors First**: Use browser dev tools to test CSS selectors
2. **Start Simple**: Begin with basic selectors and refine
3. **Use Multiple Selectors**: Provide fallback options with `|`
4. **Validate URLs**: Ensure all URLs are accessible
5. **Organize Categories**: Use meaningful category names

### Processing Strategy

1. **Start Small**: Test with a few products first
2. **Monitor Progress**: Watch logs for issues
3. **Use Appropriate Models**: Groq for speed, OpenAI for quality
4. **Set Reasonable Limits**: Don't overwhelm target sites
5. **Backup Results**: Download and archive regularly

### Performance Optimization

1. **Use Fast Models**: Groq for better speed
2. **Optimize Selectors**: Use specific, efficient selectors
3. **Batch Processing**: Process related sites together
4. **Resource Management**: Monitor system resources
5. **Error Handling**: Implement proper retry logic

## Output Organization

### File Structure

```
output/
├── Category A/
│   ├── Product 1/
│   │   ├── catalog.pdf_cleaned.pdf
│   │   ├── datasheet.pdf_cleaned.pdf
│   │   └── manual.pdf_cleaned.pdf
│   └── Product 2/
│       └── ...
├── Category B/
│   └── ...
CSVS/
├── downloaded_Category_A_20250101_120000.csv
├── downloaded_Category_B_20250101_120500.csv
└── ...
```

### CSV Reports

Each category generates a summary CSV with:
- `productLink`: URL to the product page
- `productName`: Name of the product
- `category`: Category name
- `saved_count`: Number of PDFs downloaded
- `has_datasheet`: Whether a datasheet was found

### File Naming

- **Products**: Sanitized folder names based on product names
- **PDFs**: Original filenames with `_cleaned.pdf` suffix for processed files
- **Reports**: Timestamped CSV files with category names

## Advanced Features

### Custom PDF Processing

The system automatically:
- Removes QR codes
- Cleans contact information
- Estimates background colors
- Adds cover pages
- Removes footers

### Batch Processing

Process multiple sites efficiently:
1. Create comprehensive CSV files
2. Use category organization
3. Monitor progress across all sites
4. Download consolidated results

### Integration Options

- **REST API**: Use the web interface endpoints
- **Python SDK**: Import and use programmatically
- **Command Line**: Integrate into scripts and automation
- **Desktop GUI**: Use for interactive processing

## Getting Help

### Documentation
- Check the main README for overview
- Review API Reference for technical details
- Consult Architecture docs for system understanding

### Support Channels
- GitHub Issues for bug reports
- GitHub Discussions for questions
- Code comments for implementation details

### Community
- Share configurations and tips
- Report issues and improvements
- Contribute to documentation

## Tips and Tricks

### Efficient Selectors
```css
/* Good: Specific and efficient */
.product-item .title
div[class*="product"]:not(.advertisement)

/* Avoid: Too generic */
div
*[class*="item"]
```

### Pagination Patterns
```css
/* JavaScript pagination */
button.load-more
a.next-page
button#show-more

/* URL-based pagination */
?page=1
&p=1
&offset=20
```

### PDF Selectors
```css
/* Common PDF patterns */
a[href*=".pdf"]
a[href$=".pdf"]
button[onclick*="pdf"]
.download-pdf
```

### Testing Strategies
1. **Start with one product**: Test selectors on a single page
2. **Use browser dev tools**: Inspect elements and test selectors
3. **Validate manually**: Check that selectors return expected results
4. **Scale gradually**: Increase batch size as you gain confidence
5. **Monitor results**: Always verify output quality

Remember: The key to successful crawling is understanding your target websites and crafting appropriate selectors. Take time to analyze the site structure before creating your CSV configuration.
