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
        "delay_max": 15
    },
    "pdf_settings": {
        "max_file_size_mb": 60,  # Maximum PDF file size to download
        "skip_large_files": True,  # Whether to skip files larger than max_file_size_mb
        "allowed_types": [
            "Data Sheet",
            "Technical Drawing",
            "User Manual",
            "Installation Guide",
            "Application guide",
            "Design guide",
            "User guide",
            "Generic",
            "CAD",
            "ZIP",
            "Catalog",
            "EDZ"
        ],
        "per_type_limits": {
            "Data Sheet": 2,
            "Technical Drawing": 2,
            "User Manual": 1,
            "Installation Guide": 1,
            "User guide": 1,
            "Application guide": 1,
            "Design guide": 1,
            "Generic": 1,
            "CAD": 4,
            "ZIP": 2,
            "Catalog": 1,
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