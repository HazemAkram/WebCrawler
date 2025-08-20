"""
DeepSeek AI Web Crawler
Copyright (c) 2025 Ayaz Mensyoğlu

This file is part of the DeepSeek AI Web Crawler project.
Licensed under the Apache License, Version 2.0.
See NOTICE file for additional terms and conditions.
"""



import numpy as np
import cv2
from PIL import Image
import pytesseract
from pyzbar.pyzbar import decode
from pdf2image import convert_from_path
import os
import platform
import subprocess
import shutil
import groq
from typing import List, Dict, Any
import json
from models.venue import TextRemove

# Configuration constants
QR_PADDING = 10              # Padding around QR codes
BG_SAMPLE_MARGIN = 40        # Background estimation margin
RESCALE_FACTOR = 2           # QR enhancement scale factor

# Groq API configuration
GROQ_MODEL = "openai/gpt-oss-120b"  # Default model for text analysis

def get_groq_client(api_key):
    """
    Initialize and return Groq client using API key from environment.
    Returns the client or None if API key is not available.
    """
    if not api_key:
        api_key = os.getenv('GROQ_API_KEY')
        print(f"🤖 Groq client initialized: from .env ")    
    try:
        client = groq.Groq(api_key=api_key)
        return client
    except Exception as e:
        print(f"❌ Error initializing Groq client: {str(e)}")
        return None

def analyze_text_with_ai(text_content: str, groq_client) -> List[Dict[str, Any]]:
    """
    Use Groq API to analyze text and identify contact information that should be removed.
    
    Args:
        text_content (str): The text content to analyze
        groq_client: Initialized Groq client
        
    Returns:
        List[Dict[str, Any]]: List of text regions to remove with coordinates and content
    """
    if not groq_client:
        return []
    
    prompt = f"""
You are an AI assistant specialized in identifying contact information and sensitive data in documents that should be removed for privacy and security purposes.

Analyze the following text and identify ALL instances of contact information that should be removed:

**CONTACT INFORMATION TO REMOVE:**
- Website URLs and domain names
- Physical addresses (full or partial)
- Phone numbers (including international formats)
- Fax numbers
- Email addresses
- Social media handles
- Contact person names with titles
- Company contact details
- Office locations and building information
- Postal codes and city information
- Any other identifying contact information

**TEXT TO ANALYZE:**
{text_content}

**INSTRUCTIONS:**
1. Identify each piece of contact information
2. Provide the exact text that should be removed
3. Include surrounding context if needed for accurate removal
4. Be thorough - don't miss any contact details
5. Focus on privacy and security concerns

**OUTPUT FORMAT:**
Return a JSON array containing objects with these fields:
- text_to_remove: The exact text to remove
- reason: Why this text should be removed
- confidence: Your confidence level (high/medium/low)

IMPORTANT: Return ONLY valid JSON array, no markdown formatting or additional text.

Example output format:
[
  {{
    "text_to_remove": "John Doe",
    "reason": "Contact person name",
    "confidence": "high"
  }},
  {{
    "text_to_remove": "john@example.com",
    "reason": "Email address",
    "confidence": "high"
  }}
]
"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that identifies contact information to remove from documents. Always return a valid JSON array containing objects with text_to_remove, reason, and confidence fields. Return ONLY the JSON array, no additional text or formatting."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistent results
        )
        
        # Extract the response content
        ai_response = response.choices[0].message.content.strip()
        
        # Try to parse the JSON response
        try:
            # Remove any markdown formatting if present
            if ai_response.startswith("```json"):
                ai_response = ai_response[7:]
            if ai_response.endswith("```"):
                ai_response = ai_response[:-3]
            
            ai_response = ai_response.strip()
            analysis_result = json.loads(ai_response)
            
            if isinstance(analysis_result, list):
                return analysis_result
            else:
                print(f"⚠️ Unexpected AI response format: {type(analysis_result)}")
                return []
                
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing AI response as JSON: {str(e)}")
            print(f"AI Response: {ai_response}")
            return []
            
    except Exception as e:
        print(f"❌ Error calling Groq API: {str(e)}")
        return []

def find_tesseract_path():
    """
    Automatically detect Tesseract installation path across different operating systems.
    Returns the path to tesseract executable or None if not found.
    """
    system = platform.system().lower()
    
    # Common installation paths
    common_paths = {
        'windows': [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
            r"C:\tesseract\tesseract.exe",
        ],
        'darwin': [  # macOS
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
            "/usr/bin/tesseract",
        ],
        'linux': [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/opt/tesseract/bin/tesseract",
        ]
    }
    
    # Check if tesseract is in PATH
    if shutil.which("tesseract"):
        return shutil.which("tesseract")
    
    # Check common installation paths
    paths_to_check = common_paths.get(system, [])
    for path in paths_to_check:
        if os.path.exists(path):
            return path
    
    # Try to find using system commands
    try:
        if system == 'windows':
            # Try using where command
            result = subprocess.run(['where', 'tesseract'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        else:
            # Try using which command
            result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
    except Exception:
        pass
    
    return None

def find_poppler_path():
    """
    Automatically detect Poppler installation path across different operating systems.
    Returns the path to poppler bin directory or None if not found.
    """
    system = platform.system().lower()
    
    # Common installation paths
    common_paths = {
        'windows': [
            r"C:\Program Files\Release-24.08.0-0\poppler-24.08.0\Library\bin",
            r"C:\Program Files\poppler\bin",
            r"C:\poppler\bin",
            r"C:\Program Files (x86)\poppler\bin",
        ],
        'darwin': [  # macOS
            "/usr/local/bin",
            "/opt/homebrew/bin",
            "/usr/bin",
        ],
        'linux': [
            "/usr/bin",
            "/usr/local/bin",
            "/opt/poppler/bin",
        ]
    }
    
    # Check if pdftoppm is in PATH
    if shutil.which("pdftoppm"):
        pdftoppm_path = shutil.which("pdftoppm")
        return os.path.dirname(pdftoppm_path)
    
    # Check common installation paths
    paths_to_check = common_paths.get(system, [])
    for path in paths_to_check:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "pdftoppm")):
            return path
    
    # For Windows, try to find using system commands
    if system == 'windows':
        try:
            result = subprocess.run(['where', 'pdftoppm'], capture_output=True, text=True)
            if result.returncode == 0:
                pdftoppm_path = result.stdout.strip().split('\n')[0]
                return os.path.dirname(pdftoppm_path)
        except Exception:
            pass
    
    return None

def setup_dependencies():
    """
    Setup and configure Tesseract and Poppler paths.
    Returns tuple of (tesseract_path, poppler_path) with detected or default paths.
    """
    # Detect Tesseract
    tesseract_path = find_tesseract_path()
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"✅ Tesseract found at: {tesseract_path}")
    else:
        print("⚠️ Tesseract not found. Please install Tesseract OCR:")
        print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   macOS: brew install tesseract")
        print("   Linux: sudo apt-get install tesseract-ocr")
        return None, None
    
    # Detect Poppler
    poppler_path = find_poppler_path()
    if poppler_path:
        print(f"✅ Poppler found at: {poppler_path}")
    else:
        print("⚠️ Poppler not found. Please install Poppler:")
        print("   Windows: https://github.com/oschwartz10612/poppler-windows/releases")
        print("   macOS: brew install poppler")
        print("   Linux: sudo apt-get install poppler-utils")
        return tesseract_path, None
    
    return tesseract_path, poppler_path

def test_dependencies():
    """
    Test if Tesseract and Poppler are working correctly.
    Returns True if both dependencies are working, False otherwise.
    """
    print("🔍 Testing dependencies...")
    
    # Test Tesseract
    try:
        tesseract_path, poppler_path = setup_dependencies()
        if not tesseract_path:
            print("❌ Tesseract test failed")
            return False
        
        # Test Tesseract functionality
        test_image = Image.new('RGB', (100, 100), color='white')
        result = pytesseract.image_to_string(test_image)
        print("✅ Tesseract test passed")
        
    except Exception as e:
        print(f"❌ Tesseract test failed: {str(e)}")
        return False
    
    # Test Poppler
    try:
        if not poppler_path:
            print("❌ Poppler test failed - not found")
            return False
        
        # Test if pdftoppm exists in the poppler path
        pdftoppm_path = os.path.join(poppler_path, "pdftoppm")
        if not os.path.exists(pdftoppm_path):
            print(f"❌ Poppler test failed - pdftoppm not found at {pdftoppm_path}")
            return False
        
        print("✅ Poppler test passed")
        return True
        
    except Exception as e:
        print(f"❌ Poppler test failed: {str(e)}")
        return False

def preprocess_image(img_cv):
    """Enhances image for QR detection using adaptive thresholding"""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 31, 8
    )

def enhance_qr(img_cv): 
    img_cv = cv2.resize(img_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_img = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(contrast_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 10)
    return binary

def estimate_background_color(img, bbox):
    """Estimates background color using median of area around bounding box"""
    x, y, w, h = bbox
    # Expand sampling area with margin
    x0 = max(x - BG_SAMPLE_MARGIN, 0)
    y0 = max(y - BG_SAMPLE_MARGIN, 0)
    x1 = min(x + w + BG_SAMPLE_MARGIN, img.shape[1])
    y1 = min(y + h + BG_SAMPLE_MARGIN, img.shape[0])
    
    bg_patch = img[y0:y1, x0:x1]
    return np.median(bg_patch, axis=(0, 1)) if bg_patch.size > 0 else [255, 255, 255]

def remove_region(img, bbox, padding=0):
    """Removes specified region by filling with background color"""
    x, y, w, h = bbox
    # Apply padding
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w += 2 * padding
    h += 2 * padding
    
    avg_color = estimate_background_color(img, (x, y, w, h))
    cv2.rectangle(img, (x, y), (x + w, y + h), avg_color, -1)

def replace_text_in_scanned_pdf_ai(images, api_key: str):
    """
    Automatically removes contact information using AI analysis instead of manual search text.
    Uses Groq API to identify what should be removed.
    """
    modified_images = []
    groq_client = get_groq_client(api_key)
    
    if not groq_client:
        print("⚠️ Cannot perform AI-powered text removal without Groq API key")
        print("   Returning original images unchanged")
        return images
    
    for page_num, img in enumerate(images, 1):
        print(f"🤖 Processing page {page_num} with AI...")
        
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Extract all text from the page
        data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)
        
        # Combine all text into a single string for AI analysis
        all_text = " ".join([text.strip() for text in data["text"] if text.strip()])
        
        if not all_text:
            print(f"   Page {page_num}: No text found")
            modified_images.append(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
            continue
        
        # Use AI to analyze the text and identify what should be removed
        ai_analysis = analyze_text_with_ai(all_text, groq_client)
        
        if not ai_analysis:
            print(f"   Page {page_num}: AI analysis returned no results")
            modified_images.append(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
            continue
        
        print(f"   Page {page_num}: AI identified {len(ai_analysis)} items to remove")
        
        # Process each text item to find matches with AI-identified content
        removed_count = 0
        for i, text in enumerate(data["text"]):
            text_lower = text.strip().lower()
            if not text_lower:
                continue
            if text == "www.omegamotor.com.tr":
                print(text)
            # Check if this text should be removed based on AI analysis
            for ai_item in ai_analysis:
                text_to_remove = ai_item.get("text_to_remove", "").lower()
                reason = ai_item.get("reason", "unknown")
                confidence = ai_item.get("confidence", "low")
                
                # Check if the current text contains or matches the AI-identified text
                if (text_to_remove in text_lower or 
                    text_lower in text_to_remove or 
                    text_lower == text_to_remove):
                    
                    # Get the bounding box for this text
                    bbox = (data["left"][i], data["top"][i], 
                            data["width"][i], data["height"][i])
                    
                    # Remove the text region
                    remove_region(img_cv, bbox, padding=QR_PADDING)
                    
                    print(f"      🗑️ Removed: '{text.strip()}' (Reason: {reason}, Confidence: {confidence})")
                    removed_count += 1
                    break  # Remove this text and move to next text item
        
        print(f"   Page {page_num}: Removed {removed_count} text regions")
        modified_images.append(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
    
    return modified_images

def remove_qr_codes_from_pdf(images):

    modified_images = []

    count = 1
    for img in images:
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        img_org = np.array(img)
        img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)



        processed_img = preprocess_image(img_cv)
        qr_codes = decode(processed_img)
        
        if len(qr_codes) == 0 : 
            img_enh = enhance_qr(img_cv)
            qr_codes = decode(img_enh)
            
            for qr in qr_codes:
                x, y, w, h = qr.rect  # Get bounding box of QR code
                x, y, w, h = int(x / 2), int(y / 2), int(w / 2), int(h / 2)
                    
                
                print(f" At page number {count} QR Code detected at ({x}, {y}), size ({w}x{h})")
                
                # Estimating background color
                bg_patch = img_org[max(y-40, 0):min(y+h+40, img_org.shape[0]), max(x-40, 0):min(x+w+40, img_org.shape[1])]
                avg_color = np.median(bg_patch, axis=(0, 1)) if bg_patch.size > 0 else [255, 255, 255]
                # Fill the QR code area with the estimated background color
                cv2.rectangle(img_org, (x, y), (x + w, y + h), avg_color, -1)
                
                
            # Convert back to PIL format
            modified_images.append(Image.fromarray(cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)))

        else: 
            for qr in qr_codes:
                padding = 10 
                
                x, y, w, h = qr.rect  # Get bounding box of QR code

                x = max(x - padding, 0)
                y = max(y - padding, 0)

                w = w + 2 * padding
                h = h + 2 * padding
                
                print(f" At page number {count} QR Code detected at ({x}, {y}), size ({w}x{h})")
    
                # Estimating background color
                bg_patch = img_org[max(y-40, 0):min(y+h+40, img_org.shape[0]), max(x-40, 0):min(x+w+40, img_org.shape[1])]
                avg_color = np.median(bg_patch, axis=(0, 1)) if bg_patch.size > 0 else [255, 255, 255]
    
                # Fill the QR code area with the estimated background color
                cv2.rectangle(img_org, (x, y), (x + w, y + h), avg_color, -1)
                
            # Convert back to PIL format
            modified_images.append(Image.fromarray(cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)))
            count += 1
    return modified_images



def pdf_processing(file_path: str, api_key: str):
    """
    Main processing pipeline: AI-powered text removal -> QR removal -> Save PDF
    No longer requires search_text_list parameter - fully automated with AI.
    """
    # Setup dependencies
    tesseract_path, poppler_path = setup_dependencies()
    if not tesseract_path:
        print("❌ Cannot proceed without Tesseract. Please install Tesseract OCR.")
        return
    if not poppler_path:
        print("❌ Cannot proceed without Poppler. Please install Poppler.")
        return
    
    try:
        pdf_images = convert_from_path(
            file_path,
            poppler_path=poppler_path,
        )
    except Exception as e:
        print(f"❌ Error converting PDF to images: {str(e)}")
        print("This might be due to:")
        print("1. Invalid PDF file")
        print("2. Poppler not properly installed")
        print("3. Insufficient permissions")
        return
    
    # Processing pipeline
    try:
        print("🤖 Starting AI-powered text removal...")
        text_removed = replace_text_in_scanned_pdf_ai(pdf_images, api_key)
        
        print("🔍 Starting QR code removal...")
        final_images = remove_qr_codes_from_pdf(text_removed)
        
        # Add cover page with smart resizing
        if os.path.exists("cover.png"):
            cover = Image.open("cover.png")
            
            # Get dimensions of the first page to determine PDF orientation
            if final_images:
                first_page = final_images[0]
                first_page_width, first_page_height = first_page.size
                
                # Check if PDF is horizontal (landscape orientation)
                is_horizontal = first_page_width > first_page_height
                
                if not is_horizontal:
                    # Resize cover to match first page dimensions for portrait/square PDFs
                    cover = cover.resize((first_page_width, first_page_height), Image.Resampling.LANCZOS)
                    print(f"📏 Resized cover to match PDF dimensions: {first_page_width}x{first_page_height}")
                else:
                    print(f"📏 PDF is horizontal ({first_page_width}x{first_page_height}), keeping original cover size")
                
                final_images.insert(0, cover)
            else:
                # If no pages to compare, just add cover as is
                final_images.insert(0, cover)
                print("📄 Added cover page (no pages to compare dimensions)")
        else:
            print("⚠️ cover.png not found, skipping cover page")
        
        # Check if original PDF has more than 3 pages and conditionally remove last page
        original_pdf_size = len(pdf_images)
        if original_pdf_size > 3:
            final_images = final_images  
        # Save final PDF
        final_path = f"{file_path}"
        final_images[0].save(final_path, format='PDF', save_all=True, 
                             append_images=final_images[1:])
        print(f"✨ Cleaned PDF saved to: {final_path}")
        
    except Exception as e:
        print(f"❌ Error during PDF processing: {str(e)}")
        return
    
    