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
            "Technical Manual",
            "Installation Guide",
            "Application guide",
            "Interface description",
            "Supplementary instructions",
            "Design guide",
            "User guide",
            "Selection guide",
            "Order information",
            "Brochure",
            "Generic",
            "CAD",
            "ZIP",
            "Catalog",
            "Catalogue"
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
            "Data Sheet": 1,
            "Technical Drawing": 1,
            "User Manual": 1,
            "Operating manual": 1,
            "Technical Manual": 2,
            "Installation Guide": 1,
            "Interface description": 1,
            "User guide": 1,
            "Application guide": 1,
            "Selection guide": 1,
            "Supplementary instructions": 1,
            "Design guide": 1,
            "Order information": 1,
            "Brochure": 1,
            "Generic": 1,
            "CAD": 1,
            "ZIP": 1,
            "Catalog": 1,
            "Catalogue": 1,
            "Features Catalogue": 1,
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