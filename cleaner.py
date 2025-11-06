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
from pdf2image import convert_from_path, pdfinfo_from_path
import tempfile
import img2pdf
import os
import platform
import subprocess
import shutil
import groq
from typing import List, Dict, Any
import json
from models.venue import TextRemove
import re
import gc 


# Configuration constants
QR_PADDING = 10              # Padding around QR codes (for QR removal)
TEXT_PADDING = 1             # Minimal padding around text (for text removal)
BG_SAMPLE_MARGIN = 40        # Background estimation margin
RESCALE_FACTOR = 2           # QR enhancement scale factor
OCR_CONFIDENCE_THRESHOLD = 0  # Minimum confidence for OCR text elements (0-100)

# OCR region configuration
OCR_BOTTOM_REGION_RATIO = 0.20  # Process bottom 25% of the page for OCR (0.25 = 25%)
OCR_REGION_START_RATIO = 0.60   # Start OCR processing at 75% height (1 - 0.25 = 0.75)

# Footer removal configuration
FOOTER_HEIGHT_RATIO = 0.2  # Footer height as ratio of page height (15% of page)
FOOTER_MIN_HEIGHT = 20      # Minimum footer height in pixels
FOOTER_MAX_HEIGHT = 200      # Maximum footer height in pixels
FOOTER_DETECTION_THRESHOLD = 0.3  # Threshold for detecting footer content

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

def estimate_footer_background_color(img_cv, footer_bbox):
    """
    Estimate footer background color from the footer area itself, robust to dark text.
    Prefers background pixels via Otsu threshold; falls back to median of the region.
    """
    x, y, w, h = footer_bbox
    region = img_cv[y:y+h, x:x+w]
    if region.size == 0:
        return estimate_background_color(img_cv, footer_bbox)
    try:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bg_mask = bin_img == 255  # light areas = likely background
        if np.count_nonzero(bg_mask) > 0.1 * bg_mask.size:
            bg_pixels = region[bg_mask]
            return np.median(bg_pixels, axis=0)
    except Exception:
        pass
    return np.median(region, axis=(0, 1))

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

def detect_footer_area(img_cv):
    """
    Detect footer area using multiple techniques to handle OCR-resistant text.
    Returns the optimal footer height based on content analysis.
    
    Args:
        img_cv: OpenCV image in BGR format
        
    Returns:
        int: Footer height in pixels
    """
    height, width = img_cv.shape[:2]
    
    # Calculate footer height bounds
    ratio_height = int(height * FOOTER_HEIGHT_RATIO)
    footer_height = max(FOOTER_MIN_HEIGHT, min(ratio_height, FOOTER_MAX_HEIGHT))
    
    # Method 1: Edge detection to find text boundaries
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # # Apply Gaussian blur to reduce noise
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = gray
    # Edge detection with multiple thresholds
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Focus on bottom portion of the image
    bottom_portion = int(height * 0.7)  # Bottom 30% of the image
    footer_region = edges[bottom_portion:, :]
    
    # Method 2: Horizontal projection to detect text lines
    horizontal_projection = np.sum(footer_region, axis=1)
    
    # Find the last significant text line
    threshold = np.max(horizontal_projection) * FOOTER_DETECTION_THRESHOLD
    last_text_line = -1
    
    # Scan from bottom up to find last significant content
    for i in range(len(horizontal_projection) - 1, -1, -1):
        if horizontal_projection[i] > threshold:
            last_text_line = i
            break
    
    if last_text_line > 0:
        # Add some padding below the last detected text line
        detected_footer_height = len(horizontal_projection) - last_text_line + 20
        footer_height = min(detected_footer_height, footer_height)
    
    # Method 3: Color variance analysis for footer detection
    bottom_region = img_cv[height - footer_height:, :]
    
    # Calculate color variance in the bottom region
    color_variance = np.var(bottom_region, axis=(0, 1))
    total_variance = np.sum(color_variance)
    
    # If variance is very low, it might be mostly background - reduce footer height
    if total_variance < 100:  # Threshold for low variance
        footer_height = max(FOOTER_MIN_HEIGHT, int(footer_height * 0.6))
    
    # Method 4: Text density analysis
    # Apply morphological operations to detect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(edges[height - footer_height:, :], cv2.MORPH_CLOSE, kernel)
    
    # Count non-zero pixels as text density indicator
    text_density = np.count_nonzero(morph) / (footer_height * width)
    
    # If text density is very low, reduce footer height
    if text_density < 0.01:  # Less than 1% text density
        footer_height = max(FOOTER_MIN_HEIGHT, int(footer_height * 0.5))
    
    print(f"   📏 Detected footer height: {footer_height}px (text_density: {text_density:.3f}, variance: {total_variance:.1f})")
    
    return footer_height

def remove_footer_area(img_cv, page_num, groq_client=None):
    """
    Process footer area by performing OCR and using LLM to remove contact information.
    If no text is found or no LLM client provided, removes the entire footer area.
    
    Args:
        img_cv: OpenCV image in BGR format
        page_num: Page number for logging
        groq_client: Optional Groq client for AI analysis
        
    Returns:
        OpenCV image with footer area processed
    """
    height, width = img_cv.shape[:2]
    
    # Detect optimal footer height
    footer_height = detect_footer_area(img_cv) 
    # footer_height = 200
    
    # Define footer region
    footer_start_y = height - footer_height
    footer_bbox = (0, footer_start_y, width, footer_height)
    
    # Extract footer region for OCR analysis
    footer_region = img_cv[footer_start_y:, :]
    footer_pil = Image.fromarray(cv2.cvtColor(footer_region, cv2.COLOR_BGR2RGB))
    
    # Perform OCR on footer region
    data = pytesseract.image_to_data(footer_pil, output_type=pytesseract.Output.DICT)
    
    # Check if any text was found in footer
    text_found = False
    for i in range(len(data['text'])):
        if data['text'][i].strip() and data['conf'][i] > OCR_CONFIDENCE_THRESHOLD:
            text_found = True
            break
    
    if text_found and groq_client:        
        # Create contextual text chunks for footer region
        text_chunks = create_contextual_text_chunks(data, original_image_height=footer_height, bottom_25_percent_only=False)
        
        if text_chunks:            
            # Use AI to analyze footer text chunks
            ai_analysis = analyze_text_with_ai_chunks(text_chunks, groq_client)
            
            if ai_analysis:                
                # Process AI analysis results and remove identified contact information
                removed_count = 0
                for ai_item in ai_analysis:
                    text_to_remove = ai_item.get("text_to_remove", "").lower()
                    reason = ai_item.get("reason", "unknown")
                    confidence = ai_item.get("confidence", "low")
                    
                    # Skip removal if reason is unknown
                    if reason.lower() == "unknown":
                        print(f"   ⚠️ Skipping removal of '{text_to_remove}' - reason is unknown")
                        continue
                    
                    # Find chunks that contain the text to remove
                    matching_chunks = []
                    for chunk in text_chunks:
                        chunk_text = chunk['text'].lower()
                        if not chunk_text:
                            continue
                        
                        if (text_to_remove in chunk_text or 
                            chunk_text in text_to_remove or 
                            chunk_text == text_to_remove):
                            matching_chunks.append(chunk)
                    
                    if matching_chunks:
                        # Remove matching chunks from footer region
                        for chunk in matching_chunks:
                            bbox = chunk['bbox']
                            # Adjust coordinates to footer region
                            adjusted_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
                            
                            # Remove the text region from footer
                            remove_region(footer_region, adjusted_bbox, padding=TEXT_PADDING)
                            
                            print(f"🗑️ Removed footer chunk: '{chunk['text'].strip()}' (Reason: {reason}, Confidence: {confidence})")
                            removed_count += 1
                
                print(f"   📝 Footer: Removed {removed_count} contact text chunks")
                
                # Copy processed footer back to main image
                img_cv[footer_start_y:, :] = footer_region
                print(f"   🦶 Page {page_num}: Processed footer area ({footer_height}px high) with AI contact removal")
                return img_cv
            else:
                print(f"   📝 Footer: AI found no contact information to remove")
                print(f"   🦶 Page {page_num}: Processed footer area ({footer_height}px high) with AI contact removal")
                return img_cv
        else:
            print(f"   📝 Footer: No text chunks created from OCR")
            print(f"   🦶 Page {page_num}: Processed footer area ({footer_height}px high) with AI contact removal")
            return img_cv
    else:
        if not text_found:
            print(f"   📝 No text detected in footer, removing entire area")
        else:
            print(f"   📝 No AI client provided, removing entire footer area")
    
    # Fallback: Remove entire footer area with background color
    # Estimate background color from multiple areas
    bg_samples = []
    
    # Top margin sample
    if height > 100:
        top_sample = img_cv[20:80, width//4:3*width//4]
        if top_sample.size > 0:
            bg_samples.append(np.median(top_sample, axis=(0, 1)))
    
    # Side margins sample
    if width > 200:
        left_sample = img_cv[height//4:3*height//4, 20:60]
        right_sample = img_cv[height//4:3*height//4, width-60:width-20]
        if left_sample.size > 0:
            bg_samples.append(np.median(left_sample, axis=(0, 1)))
        if right_sample.size > 0:
            bg_samples.append(np.median(right_sample, axis=(0, 1)))
    
    # Use area around footer for background estimation if no other samples
        # Estimate background strictly from the footer region
    bg_color = estimate_footer_background_color(img_cv, footer_bbox)
    
    # Apply some smoothing to avoid harsh edges
    result_img = img_cv.copy()
    
    # Create a gradient mask for smoother transition (optional)
    fade_height = min(10, footer_height // 4)  # Fade zone height
    
    if fade_height > 0 and footer_start_y - fade_height > 0:
        # Create gradient mask
        for i in range(fade_height):
            y_pos = footer_start_y - fade_height + i
            alpha = i / fade_height  # Fade from 0 to 1
            
            # Blend original color with background color
            original_line = result_img[y_pos, :].astype(np.float32)
            background_line = np.full_like(original_line, bg_color, dtype=np.float32)
            blended_line = (1 - alpha) * original_line + alpha * background_line
            result_img[y_pos, :] = blended_line.astype(np.uint8)
    
    # Fill the footer area with background color
    cv2.rectangle(result_img, (0, footer_start_y), (width, height), bg_color, -1)
    
    print(f"   🦶 Page {page_num}: Removed footer area ({footer_height}px high) with background color {bg_color}")
    
    return result_img

def kmeans(input_img, k, i_val):
    """
    Simple K-means implementation for image enhancement
    """
    hist = cv2.calcHist([input_img],[0],None,[256],[0,256])
    img = input_img.ravel()
    img = np.reshape(img, (-1, 1))
    img = img.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(img,k,None,criteria,10,flags)
    centers = np.sort(centers, axis=0)

    return centers[i_val].astype(int), centers, hist

def enhance_image_region(img_cv, start_ratio: float, end_ratio: float):
    """
    Extract a vertical band [start_ratio, end_ratio) of the image and enhance for OCR.
    Applies grayscale → light bilateral filter → CLAHE.
    Returns a PIL grayscale image suitable for Tesseract.
    
    Args:
        img_cv: OpenCV image in BGR format (numpy array)
        start_ratio: Start position as ratio of image height (0.0 to 1.0)
        end_ratio: End position as ratio of image height (0.0 to 1.0)
        
    Returns:
        PIL Image:  grayscale subregion
    """
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # gray = img_cv
    height = gray.shape[0]
    
    # Extract vertical band
    start_y = int(height * start_ratio)
    end_y = int(height * end_ratio)
    region_img = gray[start_y:end_y, :]
    
    # Apply light bilateral filter
    region_img = cv2.bilateralFilter(region_img, 5, 30, 30)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    region_img = clahe.apply(region_img)
    
    return Image.fromarray(region_img)
   
    

def replace_text_in_scanned_pdf_ai(images, api_key: str):
    """
    Automatically removes contact information using AI analysis.
    Uses K-means enhanced images for OCR, but processes original images.
    Improved approach: processes text in contextual chunks instead of word-by-word.
    
    Processing Order:
    1. OCR Analysis (bottom 25% of page) - detects readable text
    2. AI Analysis - identifies contact information from OCR results
    3. Text Removal - removes AI-identified contact information
    4. Footer Removal - handles remaining OCR-resistant text
    """
    modified_images = []
    groq_client = get_groq_client(api_key)
    
    if not groq_client:
        print("⚠️ Cannot perform AI-powered text removal without Groq API key")
        print("   Returning original images unchanged")
        return images
    
    for page_num, img in enumerate(images, 1):
        print(f"🤖 Processing page {page_num}...")
        
        # Convert to OpenCV format for processing
        # img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_cv = np.array(img)
        original_height = img_cv.shape[0]  # Store original image height for coordinate adjustment
        
        # OCR configuration
        # ocr_config = "--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./:-_"
        
        # Process page in 4 equal-height bands: [0-25%], [25-50%], [50-75%], [75-100%]
        bands = [(0.0, 0.30), (0.26, 0.60), (0.56, 0.90), (0.86, 1.0)]
        text_chunks = []
        
        for band_idx, (start_ratio, end_ratio) in enumerate(bands, 1):
            # Extract and enhance band
            enhanced_band = enhance_image_region(img_cv, start_ratio, end_ratio)
                        
            # Perform OCR on band
            data_band = pytesseract.image_to_data(
                enhanced_band,
                output_type=pytesseract.Output.DICT,
                # config=ocr_config
            )
            
            # Create contextual chunks with proper offset (no scaling, direct coordinates)
            region_offset = int(start_ratio * original_height)
            band_chunks = create_contextual_text_chunks(
                data_band,
                original_image_height=original_height,
                bottom_25_percent_only=False,
                scale=2.0,
                region_offset_top=region_offset
            )
            
            if band_chunks:
                text_chunks.extend(band_chunks)
                print(f"   Band {band_idx} [{int(start_ratio*100)}-{int(end_ratio*100)}%]: Found {len(band_chunks)} chunks")
            
            # Clean up band resources
            del enhanced_band, data_band, band_chunks
        
        gc.collect()
        
        # # testing chunks 
        # for i, chunk in enumerate(text_chunks):
        #     print(f"   Page {page_num}: Chunk {i}: {chunk['text']}")
        
        # if not text_chunks:
        #     print(f"   Page {page_num}: No text chunks found")
        #     # Still process footer area even if no OCR text was found
        #     img_cv = remove_footer_area(img_cv, page_num, groq_client)
        #     # Convert back to PIL format and add to results
        #     modified_images.append(Image.fromarray(img_cv))
        #     continue
        
        print(f"   Page {page_num}: Created {len(text_chunks)} total contextual text chunks from 4 bands")
        
        # Use AI to analyze the text chunks and identify what should be removed
        ai_analysis = analyze_text_with_ai_chunks(text_chunks, groq_client)
        
        # if not ai_analysis:
        #     print(f"   Page {page_num}: AI returned no results")
        #     # Still process footer area even if AI found nothing
        #     img_cv = remove_footer_area(img_cv, page_num, groq_client)
        #     # Convert back to PIL format and add to results
        #     modified_images.append(Image.fromarray(img_cv))
        #     continue
        
        print(f"   Page {page_num}: AI identified {len(ai_analysis)} items to remove")
        
        removed_count = 0
        
        # Process AI analysis results
        for ai_item in ai_analysis:
            text_to_remove = ai_item.get("text_to_remove", "").lower()
            reason = ai_item.get("reason", "unknown")
            confidence = ai_item.get("confidence", "low")
            chunk_ref = ai_item.get("chunk_reference", "")
            
            # Skip removal if reason is unknown
            if reason.lower() == "unknown":
                print(f"   ⚠️ Skipping removal of '{text_to_remove}' - reason is unknown")
                continue
                        
            # Find chunks that contain or match the text to remove
            matching_chunks = []
            for chunk in text_chunks:
                chunk_text = chunk['text'].lower()
                if not chunk_text:
                    continue
                
                # Check if this chunk contains or matches the AI-identified text
                if (text_to_remove in chunk_text or 
                    chunk_text in text_to_remove or 
                    chunk_text == text_to_remove):
                    matching_chunks.append(chunk)
            
            if matching_chunks:                
                # Remove all matching chunks
                for chunk in matching_chunks:
                    bbox = chunk['bbox']
                    
                    # Remove the text region from image
                    # Use minimal padding for text to fit exactly
                    remove_region(img_cv, bbox, padding=TEXT_PADDING)
                    
                    print(f"🗑️ Removed chunk: '{chunk['text'].strip()}' (Reason: {reason}, Confidence: {confidence})")
                    removed_count += 1
            else:
                print(f"   ⚠️ No matching chunks found for: '{text_to_remove}'")
        
        print(f"   Page {page_num}: Removed {removed_count} text chunks")
        
        # After OCR-based text removal, process footer area with AI (handles OCR-resistant text)
        # img_cv = remove_footer_area(img_cv, page_num, groq_client)
        
        # Convert back to PIL format and add to results
        modified_images.append(Image.fromarray(img_cv))
        del img_cv
        gc.collect()
    
    return modified_images

# New or changed function signature and internals
def create_contextual_text_chunks(data, original_image_height=None, bottom_25_percent_only=True, scale: float = 1.0, region_offset_top: int = None):
    """
    Create contextual text chunks from Tesseract output data.
    Groups related text elements into meaningful chunks for better AI analysis.

    Args:
        data: Tesseract output data dictionary
        original_image_height: Height of the original image (needed to adjust coordinates for bottom 25% processing)
        bottom_25_percent_only: If True, adjust coordinates for bottom 25% region processing (ignored if region_offset_top is set)
        scale: upscale factor applied to the OCR image (e.g., 2.0 when fx=fy=2 in enhance_image_region)
        region_offset_top: If provided, use this as the y-offset directly (overrides bottom_25_percent_only logic)
    """
    if not data or 'text' not in data:
        return []
    
    chunks = []
    current_chunk = None
    
    # Calculate offset for region (in original coordinates)
    y_offset = 0
    if region_offset_top is not None:
        # Use explicit offset provided by caller
        y_offset = region_offset_top
    elif bottom_25_percent_only and original_image_height:
        # Legacy behavior: calculate offset for bottom 25% region
        y_offset = int(original_image_height * OCR_REGION_START_RATIO)

    text_elements = []
    def _to_int(val, default=0):
        try:
            return int(float(val))
        except Exception:
            return default

    # Map OCR coordinates (in upscaled space) back to original image space
    sx = float(scale) if scale else 1.0
    sy = float(scale) if scale else 1.0

    for i in range(len(data['text'])):
        if str(data['text'][i]).strip():
            left_up = _to_int(data.get('left', [0])[i])
            top_up = _to_int(data.get('top', [0])[i])
            width_up = _to_int(data.get('width', [0])[i])
            height_up = _to_int(data.get('height', [0])[i])
            conf = _to_int(data.get('conf', [0])[i], default=-1)

            # Downscale to original image coordinates and add region offset
            left = int(round(left_up / sx))
            top = int(round(top_up / sy)) + y_offset
            width = int(round(width_up / sx))
            height = int(round(height_up / sy))

            text_elements.append({
                'text': str(data['text'][i]).strip(),
                'left': left,
                'top': top,
                'width': width,
                'height': height,
                'conf': conf
            })
    
    # Sort and chunking logic unchanged...
    text_elements.sort(key=lambda x: (x['top'], x['left']))
    for element in text_elements:
        if element['conf'] < OCR_CONFIDENCE_THRESHOLD:
            continue

        if current_chunk is None:
            current_chunk = {
                'text': element['text'],
                'bbox': (element['left'], element['top'], element['width'], element['height']),
                'elements': [element]
            }
        else:
            should_merge = should_merge_text_elements(current_chunk, element)
            if should_merge:
                current_chunk['text'] += ' ' + element['text']
                current_chunk['bbox'] = merge_bounding_boxes(
                    current_chunk['bbox'],
                    (element['left'], element['top'], element['width'], element['height'])
                )
                current_chunk['elements'].append(element)
            else:
                if current_chunk['text'].strip():
                    chunks.append(current_chunk)
                current_chunk = {
                    'text': element['text'],
                    'bbox': (element['left'], element['top'], element['width'], element['height']),
                    'elements': [element]
                }

    if current_chunk and current_chunk['text'].strip():
        chunks.append(current_chunk)
    return chunks

def should_merge_text_elements(chunk, element):
    """
    Determine if a text element should be merged with the current chunk.
    Uses spatial and contextual rules for intelligent grouping.
    
    Args:
        chunk: Current text chunk
        element: Text element to consider for merging
        
    Returns:
        bool: True if elements should be merged
    """
    # Get current chunk's right and bottom boundaries
    chunk_right = chunk['bbox'][0] + chunk['bbox'][2]
    chunk_bottom = chunk['bbox'][1] + chunk['bbox'][3]
    
    # Check horizontal proximity (same line)
    horizontal_distance = abs(element['left'] - chunk_right)
    vertical_distance = abs(element['top'] - chunk['bbox'][1])
    
    # Same line: small vertical distance, reasonable horizontal distance
    if vertical_distance <= 20:  # Within 20 pixels vertically
        if horizontal_distance <= 50:  # Within 50 pixels horizontally
            return True
    
    # Check vertical proximity (next line)
    if horizontal_distance <= 30:  # Roughly aligned horizontally
        if 0 < vertical_distance <= 40:  # Within 40 pixels below
            return True
    
    # Check if this looks like a continuation (e.g., phone number parts)
    current_text = chunk['text'].lower()
    new_text = element['text'].lower()
    
    # Common patterns that should be merged
    merge_patterns = [
        # Phone numbers
        (r'\d{3,4}$', r'^\d{3,4}'),  # Area code + number
        (r'\d{3}$', r'^\d{4}'),       # 3 digits + 4 digits
        (r'\d{4}$', r'^\d{4}'),       # 4 digits + 4 digits
        (r'\(\d{3}\)$', r'^\d{3}'),   # (555) + 123
        (r'\d{3}$', r'^\d{3}-\d{4}'), # 555 + 123-4567
        
        # Addresses
        (r'street$|st\.$|avenue$|ave\.$|road$|rd\.$|drive$|dr\.$|lane$|ln\.$|boulevard$|blvd\.$', r'^\d+'),  # Street name + number
        (r'\d+$', r'^[A-Z][a-z]+'),   # Number + street name
        (r'[A-Z][a-z]+$', r'^\d+'),   # Street name + number
        (r'\d+$', r'^[A-Z][a-z]+\s+[A-Z][a-z]+'),  # Number + "Main Street"
        
        # Names
        (r'^[A-Z][a-z]+$', r'^[A-Z][a-z]+$'),  # First + Last name
        (r'[A-Z][a-z]+$', r'^[A-Z][a-z]+'),    # Middle + Last name
        
        # Company names
        (r'inc\.$|corp\.$|llc$|ltd\.$|company$|co\.$|corporation$', r'^[A-Z]'),  # Company suffix + continuation
        (r'[A-Z][a-z]+$', r'^&'),     # Company name + "& Associates"
        
        # Email addresses
        (r'[a-zA-Z0-9._%+-]+$', r'^@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),  # Username + @domain
        (r'[a-zA-Z0-9._%+-]+@$', r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),  # Username@ + domain
        
        # URLs
        (r'https?://$', r'^[a-zA-Z0-9.-]+'),  # http:// + domain
        (r'www\.$', r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),  # www. + domain
        
        # Postal codes
        (r'[A-Z]\d[A-Z]$', r'^\d[A-Z]\d'),  # Canadian postal code format
        (r'\d{5}$', r'^-\d{4}'),      # US ZIP + 4 format
    ]
    
    for pattern1, pattern2 in merge_patterns:
        if re.search(pattern1, current_text) and re.search(pattern2, new_text):
            return True
    
    return False

def merge_bounding_boxes(bbox1, bbox2):
    """
    Merge two bounding boxes into a single encompassing box.
    
    Args:
        bbox1: First bounding box (left, top, width, height)
        bbox2: Second bounding box (left, top, width, height)
        
    Returns:
        tuple: Merged bounding box (left, top, width, height)
    """
    left1, top1, width1, height1 = bbox1
    left2, top2, width2, height2 = bbox2
    
    # Calculate new boundaries
    new_left = min(left1, left2)
    new_top = min(top1, top2)
    new_right = max(left1 + width1, left2 + width2)
    new_bottom = max(top1 + height1, top2 + height2)
    
    return (new_left, new_top, new_right - new_left, new_bottom - new_top)

def analyze_text_with_ai_chunks(text_chunks, groq_client):
    """
    Use Groq API to analyze text chunks and identify contact information that should be removed.
    Improved version that works with contextual text chunks instead of raw text.
    
    Args:
        text_chunks: List of text chunks with bounding box information
        groq_client: Initialized Groq client
        
    Returns:
        List[Dict[str, Any]]: List of text regions to remove with coordinates and content
    """
    if not groq_client or not text_chunks:
        return []
    
    # Prepare text chunks for AI analysis
    chunk_texts = []
    for i, chunk in enumerate(text_chunks):
        chunk_texts.append(f"Chunk {i+1}: '{chunk['text']}'")
    
    analysis_text = "\n".join(chunk_texts)

    
    prompt = f"""
        You are an AI assistant specialized in identifying contact information that must be removed from documents.

        Your task: Analyze the following text chunks and return ALL contact information to remove. Always include the full phrase that a human would consider the contact detail, not just the raw value.

        CONTACT INFORMATION TO REMOVE (include label + value when present):
        - Emails (remove: label + separator + address), e.g., "Email: sales@company.org", "E-mail - info@acme.com", "info@sika.net"
        - Emails (remove: address), e.g., "info@sika.net" [the OCR may fall and read @ as a Q letter or any other letter]
        - Phone numbers (Tel/Phone/Mobile/GSM/WhatsApp/etc.; remove label + value), e.g., "Tel: +1 555 123 4567", "Mobile - 0505 123 45 67"
        - Fax numbers (remove label + value), e.g., "Fax: +44 20 1234 5678"
        - URLs/domains (remove label + value if label exists), e.g., "Website: www.acme.com", "www.acme.com"
        - Physical addresses (remove full address line(s) including city/region/postal code/building where present)
        - conturry and cities names (e.g. X-Cel House, Chrysalis Way, Langley Bridge, Eastwood, Nottinghamshire NG16 3RY)
        - Contact person names with roles/titles when tied to contact lines (e.g., "John Doe, Sales Manager")
        - Company contact blocks, office locations
        - Postal codes and city info when part of an address line

        LABELS AND SEPARATORS:
        - Include preceding labels and separators when present: "Regiested", "Email", "E-mail", "Mail", "Tel", "Telephone", "Phone", "Mobile", "GSM", "WhatsApp", "Whatsapp", "Fax", "Web", "Website", "Site", "Adres", "Address", "Firma", "Ofis", "Şube", "Telefon", "E-posta".
        - Include adjacent punctuation/separators like ':', '-', '—', '|' and surrounding spaces that belong to the phrase.

        CHUNKING RULES:
        - Prefer returning text exactly as it appears in a single chunk (label + value together).
        - If the contact information spans multiple chunks (e.g., label in one chunk, value in the next), return SEPARATE objects for each chunk’s exact text instead of merging into one string. Use chunk references to show they belong together.
        - If an address spans multiple lines/chunks, return one object per line/chunk (do not invent text that isn't present).

        OUTPUT FORMAT:
        Return ONLY a JSON array of objects with:
        - text_to_remove: exact text to remove as it appears (include label + value if in the same chunk; include separators/punctuation)
        - reason: why it should be removed (email, phone, fax, url, address, contact name, etc.)
        - confidence: high/medium/low
        - chunk_reference: "Chunk N" or "Chunks N-M" that contain the text

        TEXT CHUNKS TO ANALYZE:
        {analysis_text}

        IMPORTANT:
        - Do not return markdown, only a valid JSON array.
        - Be exhaustive: capture all instances.
        - Do not return values without their labels if the label is in the same chunk. If label and value are in different chunks, return them as separate objects (one per chunk).
"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that identifies contact information to remove from documents. Always return a valid JSON array containing objects with text_to_remove, reason, confidence, and chunk_reference fields. Return ONLY the JSON array, no additional text or formatting."
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

        del img_cv, img_org, processed_img, qr_codes
        gc.collect()
    return modified_images


def pdf_processing(file_path: str, api_key: str, log_callback=None):
    """
    Main processing pipeline: AI-powered text removal -> QR removal -> Save PDF
    No longer requires search_text_list parameter - fully automated with AI.
    
    Args:
        file_path (str): Path to the PDF file to process
        api_key (str): API key for AI processing
        log_callback (function): Optional callback function for logging to web interface
    """
    def log_message(message, level="INFO"):
        """Helper function to log messages either to callback or console"""
        if log_callback:
            log_callback(message, level)
        else:
            print(f"[{level}] {message}")
    
    # Setup dependencies
    tesseract_path, poppler_path = setup_dependencies()
    if not tesseract_path:
        log_message("❌ Cannot proceed without Tesseract. Please install Tesseract OCR.", "ERROR")
        return
    if not poppler_path:
        log_message("❌ Cannot proceed without Poppler. Please install Poppler.", "ERROR")
        return
    
    info = pdfinfo_from_path(file_path, poppler_path=poppler_path)
    page_count = int(info.get("Pages", 1))

    target_dpi = 300 
    if page_count >= 30: 
        target_dpi = 200
    elif page_count >=80: 
        target_dpi = 150


    try:
        pdf_images = convert_from_path(
            file_path,
            poppler_path=poppler_path,
            dpi=target_dpi
        )
    except Exception as e:
        log_message(f"❌ Error converting PDF to images: {str(e)}", "ERROR")
        return
    
    # Processing pipeline
    try:
        print("🤖 Starting text removal...")
        text_removed = replace_text_in_scanned_pdf_ai(pdf_images, api_key)
        log_message("✅ Text removal completed", "INFO")
        
        print("🔍 Starting QR code removal...")
        final_images = remove_qr_codes_from_pdf(text_removed)
        log_message("✅ QR code removal completed", "INFO")
        
        # Add cover page with smart resizing
        if os.path.exists("cover.png"):
            cover = Image.open("cover.png")
            # Convert cover to RGB if needed (JPEG doesn't support palette mode)
            if cover.mode != "RGB":
                cover = cover.convert("RGB")
            
            # Get dimensions of the first page to determine PDF orientation
            if final_images:
                first_page = final_images[0]
                first_page_width, first_page_height = first_page.size
                
                # Check if PDF is horizontal (landscape orientation)
                is_horizontal = first_page_width > first_page_height
                
                if not is_horizontal:
                    # Resize cover to match first page dimensions for portrait/square PDFs
                    cover = cover.resize((first_page_width, first_page_height), Image.Resampling.LANCZOS)
                                
                final_images.insert(0, cover)
            else:
                # If no pages to compare, just add cover as is
                final_images.insert(0, cover)
        else:
            log_message("⚠️ cover.png not found, skipping cover page", "WARNING")
        
        # Check if original PDF has more than 6 pages and conditionally remove last page
        original_pdf_size = len(pdf_images)
        if original_pdf_size > 6:
            final_images = final_images[:-1] 
        # Save final PDF
        final_path = f"{file_path}"
        

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_files = []
            for idx, im in enumerate(final_images):
                # Convert to RGB if needed (JPEG doesn't support palette mode or other modes)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                tmp_path = os.path.join(tmpdir, f"page_{idx:05d}.jpg")
                im.save(tmp_path, format="JPEG", quality=85, optimize=True)
                temp_files.append(tmp_path)
                # free per-page PIL object ASAP
                final_images[idx] = None
            # combine with img2pdf reading from disk (low RAM)
            with open(final_path, "wb") as f_out:
                f_out.write(img2pdf.convert(temp_files))
        print(f"✨ Cleaned PDF saved to: {final_path}")
        
    except Exception as e:
        log_message(f"❌ Error during PDF processing: {str(e)}", "ERROR")
        return
    
    