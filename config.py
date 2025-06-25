REQUIRED_KEYS = [
    "productName",
    "productLink",
]

# Pagination configuration for different websites
PAGINATION_CONFIGS = {
    # Default configuration
    "default": {
        "max_pages": 50,
        "max_consecutive_empty": 3,
        "delay_between_pages": (2, 5),
        "delay_between_links": (5, 10),
        "delay_between_pdfs": (1, 3),
    },
    
    # E-commerce platforms
    "shopify": {
        "max_pages": 100,
        "max_consecutive_empty": 2,
        "delay_between_pages": (1, 3),
        "delay_between_links": (3, 7),
        "delay_between_pdfs": (1, 2),
        "pagination_patterns": ["page", "p"],
    },
    
    "woocommerce": {
        "max_pages": 80,
        "max_consecutive_empty": 3,
        "delay_between_pages": (2, 4),
        "delay_between_links": (4, 8),
        "delay_between_pdfs": (1, 3),
        "pagination_patterns": ["paged", "page"],
    },
    
    "magento": {
        "max_pages": 60,
        "max_consecutive_empty": 2,
        "delay_between_pages": (2, 5),
        "delay_between_links": (5, 10),
        "delay_between_pdfs": (2, 4),
        "pagination_patterns": ["p", "page"],
    },
    
    # Manufacturer websites
    "manufacturer": {
        "max_pages": 40,
        "max_consecutive_empty": 3,
        "delay_between_pages": (3, 6),
        "delay_between_links": (6, 12),
        "delay_between_pdfs": (2, 5),
        "pagination_patterns": ["page", "offset", "start"],
    },
    
    # Marketplaces
    "marketplace": {
        "max_pages": 30,
        "max_consecutive_empty": 2,
        "delay_between_pages": (1, 3),
        "delay_between_links": (3, 6),
        "delay_between_pdfs": (1, 2),
        "pagination_patterns": ["page", "offset"],
    },
}

# Website-specific configurations
WEBSITE_CONFIGS = {
    "omegamotor.com.tr": {
        "type": "manufacturer",
        "custom_patterns": ["page"],
        "no_results_patterns": ["No Results Found"],
    },
    "example-shop.com": {
        "type": "shopify",
        "custom_patterns": ["page", "p"],
        "no_results_patterns": ["No products found"],
    },
    "another-store.com": {
        "type": "woocommerce",
        "custom_patterns": ["paged"],
        "no_results_patterns": ["No products were found"],
    },
}

# Crawling behavior settings
CRAWLING_SETTINGS = {
    "retry_failed_pages": True,
    "max_retries": 3,
    "retry_delay": (5, 15),
    "respect_robots_txt": True,
    "follow_redirects": True,
    "timeout": 30,
    "max_concurrent_requests": 1,  # Be respectful to servers
}

# PDF download settings
PDF_SETTINGS = {
    "max_file_size_mb": 50,
    "allowed_extensions": [".pdf"],
    "skip_existing": True,
    "create_cleaned_versions": True,
    "add_cover_page": True,
    "remove_qr_codes": True,
    "remove_specific_text": True,
}
