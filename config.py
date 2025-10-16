"""
DeepSeek AI Web Crawler
Copyright (c) 2025 Ayaz Mensyoğlu

This file is part of the DeepSeek AI Web Crawler project.
Licensed under the Apache License, Version 2.0.
See NOTICE file for additional terms and conditions.
"""


# Required keys for product extraction (category pages now return only links)
REQUIRED_KEYS = [
    "productLink",
]

PDF_KEYS = [
    "url",
    "text",
    "type",
    "language",
    "priority",
    "productName",
]

TEXT_REMOVE_KEYS = [
    "text_to_remove",
    "reason",
    "confidence",
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
        "max_file_size_mb": 60,  # Maximum PDF file size to download
        "skip_large_files": True,  # Whether to skip files larger than max_file_size_mb
        "allowed_types": [
            "Data Sheet",
            "Technical Drawing",
            "User Manual",
            "CAD",
            "ZIP",
            "Catalog",
            "EDZ"
        ],
        "per_type_limits": {
            "Data Sheet": 1,
            "Technical Drawing": 2,
            "User Manual": 1,
            "CAD": 4,
            "ZIP": 2,
            "Catalog": 1,
            "EDZ": 1
        }
    },
    "concurrency": {
        "max_concurrent_sites": 4,  # Maximum number of sites to process in parallel
        "max_products_per_site": 8,  # Maximum concurrent products per site
        "max_concurrent_downloads": 16,  # Maximum concurrent file downloads
        "max_concurrent_pdf_clean": None,  # Maximum concurrent PDF cleaning jobs (None = auto-detect)
        "max_browser_pool_per_worker": 2,  # Maximum browser instances per worker
        "per_domain_limit": 2,  # Maximum concurrent requests per domain
        "enable_fast_category_regex": False,  # Use regex extraction for category pages (skip LLM)
        "request_delay_jitter_ms": (100, 400),  # Random delay range in milliseconds
        "retry_max_attempts": 3,  # Maximum retry attempts for failed requests
        "retry_backoff_base": 2,  # Exponential backoff base multiplier
        "browser_recycle_after_products": 50,  # Recycle browser after N products
        "circuit_breaker_threshold": 5,  # Failed requests before circuit breaker trips
        "circuit_breaker_timeout": 60,  # Seconds to wait before retrying failed domain
    }
}

# Environment variable names
ENV_VARS = {
    "GROQ_API_KEY": "GROQ_API_KEY",
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY"
}