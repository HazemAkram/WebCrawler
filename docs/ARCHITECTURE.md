# Architecture Documentation

## System Overview

The DeepSeek AI Web Crawler is built on a modular architecture that separates concerns and provides flexibility for different use cases. The system is designed to be scalable, maintainable, and extensible.

## Core Components

### 1. Web Crawling Engine
- **Framework**: Crawl4AI
- **Browser**: Chromium via Playwright
- **Strategy**: Async/await pattern for concurrent processing
- **Configuration**: Customizable browser settings and timeouts

### 2. AI Processing Layer
- **LLM Integration**: Multiple providers (Groq, OpenAI, Anthropic)
- **Extraction Strategy**: Structured data extraction using Pydantic models
- **Content Analysis**: AI-powered PDF content understanding
- **Fallback Mechanisms**: Regex-based extraction when AI fails

### 3. PDF Processing Pipeline
- **Image Conversion**: PDF to image conversion using Poppler
- **OCR Processing**: Text recognition using Tesseract
- **AI Analysis**: Contact information and QR code detection
- **Content Cleaning**: Background estimation and content removal

### 4. Data Management
- **Input Validation**: CSV configuration validation
- **Output Organization**: Structured file and folder hierarchy
- **Progress Tracking**: Real-time status updates
- **Error Handling**: Graceful degradation and recovery

## Data Flow Architecture

```
CSV Input → Configuration Parser → Web Crawler → LLM Extractor → Product Data
                                                                      ↓
PDF Links → PDF Downloader → Image Converter → AI Processor → Cleaned PDF
                                                                      ↓
File Organizer → CSV Reports → Archive Manager → Final Output
```

## Module Structure

### Core Modules

#### `main.py`
- **Purpose**: Main crawling orchestration
- **Responsibilities**:
  - CSV parsing and validation
  - Crawling session management
  - Progress tracking and callbacks
  - Error handling and recovery

#### `app.py`
- **Purpose**: Web interface and API
- **Responsibilities**:
  - Flask web server
  - Real-time progress updates
  - File upload and management
  - Security and authentication

#### `cleaner.py`
- **Purpose**: PDF processing and content cleaning
- **Responsibilities**:
  - QR code detection and removal
  - Contact information identification
  - Background color estimation
  - Image processing and enhancement

### Utility Modules

#### `utils/scraper_utils.py`
- **Purpose**: Web scraping utilities and pagination
- **Key Functions**:
  - Browser configuration management
  - LLM strategy setup
  - Pagination handling
  - PDF link extraction

#### `utils/data_utils.py`
- **Purpose**: Data validation and processing
- **Key Functions**:
  - Duplicate detection
  - Data completeness validation
  - Schema validation

#### `models/venue.py`
- **Purpose**: Data models and schemas
- **Classes**:
  - `Venue`: Product data structure
  - `PDF`: PDF metadata structure
  - `TextRemove`: Text removal instructions

## Configuration Architecture

### Environment Variables
```env
GROQ_API_KEY=your_api_key
OPENAI_API_KEY=your_api_key
ANTHROPIC_API_KEY=your_api_key
FLASK_SECRET_KEY=your_secret_key
FLASK_APP_PASSWORD=your_password
```

### Configuration Files

#### `config.py`
- Default settings and constants
- Model configurations
- Crawler parameters
- PDF processing options

#### CSV Configuration
- Site-specific crawling parameters
- CSS selectors for data extraction
- Pagination configuration
- Output organization settings

## Security Architecture

### Authentication
- Optional password protection for web interface
- Secure session management
- API key validation and rotation

### Data Protection
- Path traversal prevention
- Input sanitization
- Secure file handling
- No persistent data storage

### Privacy
- Local processing only
- Temporary file cleanup
- No data transmission to third parties
- Configurable logging levels

## Performance Architecture

### Concurrency
- Async/await pattern throughout
- Concurrent PDF processing
- Parallel page crawling
- Non-blocking I/O operations

### Resource Management
- Browser session recycling
- Memory cleanup and garbage collection
- Configurable timeouts and limits
- Graceful error recovery

### Scalability
- Modular design for easy extension
- Configurable batch sizes
- Horizontal scaling support
- Load balancing capabilities

## Error Handling Architecture

### Exception Hierarchy
- Custom exception classes
- Graceful degradation
- Retry mechanisms
- Fallback strategies

### Logging System
- Multi-level logging (DEBUG, INFO, WARNING, ERROR)
- Structured log messages
- Real-time progress updates
- Error tracking and reporting

### Recovery Mechanisms
- Automatic retry with exponential backoff
- Session recovery and cleanup
- Partial result preservation
- Resume capability for long-running operations

## Integration Points

### External Services
- **Groq API**: Fast LLM inference
- **OpenAI API**: High-quality text processing
- **Anthropic API**: Balanced performance models

### System Dependencies
- **Tesseract OCR**: Text recognition
- **Poppler**: PDF processing
- **Playwright**: Browser automation

### File System
- **Input**: CSV configuration files
- **Output**: Organized PDFs and reports
- **Temporary**: Processing artifacts
- **Archives**: Compressed results

## Extension Points

### Custom Extractors
- Implement custom data extraction logic
- Add new LLM providers
- Create specialized processing pipelines

### Custom Processors
- Add new PDF processing steps
- Implement custom content filters
- Create specialized output formats

### Custom Interfaces
- Build new user interfaces
- Add API endpoints
- Create integration adapters

## Deployment Architecture

### Development
- Local Python environment
- Docker support available
- Hot reloading for development
- Comprehensive testing suite

### Production
- Scalable web server deployment
- Load balancing support
- Monitoring and alerting
- Backup and recovery procedures

### Cloud Deployment
- Container orchestration support
- Auto-scaling capabilities
- Cloud storage integration
- Managed service compatibility

## Monitoring and Observability

### Metrics
- Processing throughput
- Error rates and types
- Resource utilization
- Performance benchmarks

### Logging
- Structured log output
- Real-time monitoring
- Historical analysis
- Debug information

### Health Checks
- System status endpoints
- Dependency verification
- Performance monitoring
- Alert configuration

## Future Architecture Considerations

### Planned Enhancements
- Microservices architecture
- Event-driven processing
- Advanced caching strategies
- Machine learning integration

### Scalability Improvements
- Distributed processing
- Message queue integration
- Database persistence
- API rate limiting

### Security Enhancements
- OAuth integration
- Role-based access control
- Audit logging
- Compliance features
