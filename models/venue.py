"""
DeepSeek AI Web Crawler
Copyright (c) 2025 Ayaz Mensyoğlu

This file is part of the DeepSeek AI Web Crawler project.
Licensed under the Apache License, Version 2.0.
See NOTICE file for additional terms and conditions.
"""


from pydantic import BaseModel


class Venue(BaseModel):
    """
    Represents the data structure of a Venue.
    """

    productName: str
    productLink: str


class PDF(BaseModel):
    """
    Represents the data structure of a PDF.
    """

    url: str
    text: str
    type: str
    language: str
    priority: str
