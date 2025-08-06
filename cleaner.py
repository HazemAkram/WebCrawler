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
from fuzzywuzzy import fuzz  # For fuzzy text matching
import os
import platform
import subprocess
import shutil

# Configuration constants
TEXT_MATCH_THRESHOLD = 80    # Fuzzy match threshold (0-100)
QR_PADDING = 10              # Padding around QR codes
BG_SAMPLE_MARGIN = 40        # Background estimation margin
RESCALE_FACTOR = 2           # QR enhancement scale factor

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

def replace_text_in_scanned_pdf(search_text_list, images):
    """Removes matching text using fuzzy matching and background filling"""
    modified_images = []
    
    for page_num, img in enumerate(images, 1):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)
        
        for i, text in enumerate(data["text"]):
            text_lower = text.strip().lower()
            # Check against all search terms in the list
            for search_text in search_text_list:
                search_lower = search_text.lower()
                if fuzz.ratio(text_lower, search_lower) >= TEXT_MATCH_THRESHOLD:
                    bbox = (data["left"][i], data["top"][i], 
                            data["width"][i], data["height"][i])
                    remove_region(img_cv, bbox, padding=QR_PADDING)
                    print(f"Removed text at page {page_num}: {text.strip()} (matched: {search_text})")
                    break  # Remove this text and move to next text item

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



def pdf_processing(search_text_list: list, file_path: str):
    """Main processing pipeline: Text removal -> QR removal -> Save PDF"""
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
        text_removed = replace_text_in_scanned_pdf(search_text_list, pdf_images)
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
            final_images = final_images[:-1]  
        # Save final PDF
        final_path = f"{file_path}"
        final_images[0].save(final_path, format='PDF', save_all=True, 
                             append_images=final_images[1:])
        print(f"✨ Cleaned PDF saved to: {final_path}")
        
    except Exception as e:
        print(f"❌ Error during PDF processing: {str(e)}")
        return
    
    