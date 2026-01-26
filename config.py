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
        "delay_max": 15,
        # Safety limits for JS-based pagination / load-more behavior
        # These can be overridden via environment variables:
        #   MAX_JS_PAGINATION_PAGES, MAX_JS_PAGINATION_CLICKS
        "max_js_pagination_pages": 500,
        "max_js_pagination_clicks": 100,
    },
    "pdf_settings": {
        "max_file_size_mb": 300,  # Maximum PDF file size to download
        "skip_large_files": True,  # Whether to skip files larger than max_file_size_mb
        "allowed_types": [
            "Data Sheet",
            "Technical Drawing",
            "User Manual",
            "Operating manual",
            "Installation Guide",
            "Application guide",
            "Interface description",
            "Supplementary instructions",
            "Design guide",
            "User guide",
            "Order information",
            "Brochure",
            "Generic",
            "CAD",
            "ZIP",
            "Catalog",
            "Features Catalogue"
            "EDZ"
        ],
        "disallowed_types": [
            "Demo Version",
            "Free version",
            "30 days Test Version",
            "Declaration of conformity",
            "Press release",
            "Certificate",
        ],
        "per_type_limits": {
            "Data Sheet": 2,
            "Technical Drawing": 4,
            "User Manual": 2,
            "Operating manual": 2,
            "Installation Guide": 2,
            "Interface description": 2,
            "User guide": 2,
            "Application guide": 2,
            "Supplementary instructions": 2,
            "Design guide": 2,
            "Order information": 2,
            "Brochure": 2,
            "Generic": 3,
            "CAD": 2,
            "ZIP": 1,
            "Catalog": 2,
            "Features Catalogue": 2,
            "EDZ": 1
        }
    }
}

# Environment variable names
ENV_VARS = {
    "GROQ_API_KEY": "GROQ_API_KEY",
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY"
}