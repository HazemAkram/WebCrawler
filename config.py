"""
DeepSeek AI Web Crawler
Copyright (c) 2025 Ayaz Mensyoğlu

This file is part of the DeepSeek AI Web Crawler project.
Licensed under the Apache License, Version 2.0.
See NOTICE file for additional terms and conditions.
"""


# Required keys for product extraction
REQUIRED_KEYS = [
    "productName",
    "productLink",
]

# Default configuration settings
DEFAULT_CONFIG = {
    "output_folder": "output",
    "default_model": "groq/deepseek-r1-distill-llama-70b",
    "available_models": [
        "groq/deepseek-r1-distill-llama-70b",
        "groq/llama3-8b-8192",
        "groq/llama3-70b-8192",
        "groq/mixtral-8x7b-32768",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic/claude-3-haiku-20240307"
    ],
    "crawler_settings": {
        "page_timeout": 30000,
        "max_pages": 10,
        "delay_min": 3,
        "delay_max": 15
    }
}

# Environment variable names
ENV_VARS = {
    "GROQ_API_KEY": "GROQ_API_KEY",
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY"
}