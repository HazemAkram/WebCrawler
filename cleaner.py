import numpy as np
import cv2
from PIL import Image
import pytesseract
from pyzbar.pyzbar import decode
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz  # For fuzzy text matching

# Configuration constants
TEXT_MATCH_THRESHOLD = 80    # Fuzzy match threshold (0-100)
QR_PADDING = 10              # Padding around QR codes
BG_SAMPLE_MARGIN = 40        # Background estimation margin
RESCALE_FACTOR = 2           # QR enhancement scale factor

def preprocess_image(img_cv):
    """Enhances image for QR detection using adaptive thresholding"""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 31, 8
    )

def enhance_qr(img_cv):
    """Improves QR detection through scaling and contrast enhancement"""
    img_cv = cv2.resize(img_cv, None, fx=RESCALE_FACTOR, fy=RESCALE_FACTOR, 
                        interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray)
    return cv2.adaptiveThreshold(
        contrast_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 10
    )

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

def replace_text_in_scanned_pdf(search_text, images):
    """Removes matching text using fuzzy matching and background filling"""
    modified_images = []
    search_lower = search_text.lower()
    
    for page_num, img in enumerate(images, 1):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)
        
        for i, text in enumerate(data["text"]):
            if fuzz.ratio(text.strip().lower(), search_lower) >= TEXT_MATCH_THRESHOLD:
                bbox = (data["left"][i], data["top"][i], 
                        data["width"][i], data["height"][i])
                remove_region(img_cv, bbox, padding=QR_PADDING)
                print(f"Removed text at page {page_num}: {text.strip()}")

        modified_images.append(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
    
    return modified_images

def remove_qr_codes_from_pdf(images):
    """Detects and removes QR codes using unified processing logic"""
    modified_images = []
    
    for page_num, img in enumerate(images, 1):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_org = img_cv.copy()
        
        # First detection attempt
        processed_img = preprocess_image(img_cv)
        qr_codes = decode(processed_img)
        scale_factor = 1.0
        
        # Second attempt with enhanced image
        if not qr_codes:
            enhanced_img = enhance_qr(img_cv)
            qr_codes = decode(enhanced_img)
            scale_factor = 1/RESCALE_FACTOR
        
        # Process detected QR codes
        for qr in qr_codes:
            # Scale coordinates to original image size
            x = int(qr.rect.left * scale_factor)
            y = int(qr.rect.top * scale_factor)
            w = int(qr.rect.width * scale_factor)
            h = int(qr.rect.height * scale_factor)
            
            print(f"QR detected at page {page_num}: ({x}, {y}) [{w}x{h}]")
            remove_region(img_org, (x, y, w, h), padding=QR_PADDING)
        
        modified_images.append(Image.fromarray(cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)))
    
    return modified_images

def pdf_processing(search_text: str, file_path: str):
    """Main processing pipeline: Text removal -> QR removal -> Save PDF"""
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    print(f'Converting {file_path} to images...')
    pdf_images = convert_from_path(
        file_path,
        poppler_path=r"C:\Program Files\Release-24.08.0-0\poppler-24.08.0\Library/bin",
        )
    print('Conversion complete')
    
    # Processing pipeline
    text_removed = replace_text_in_scanned_pdf(search_text, pdf_images)
    final_images = remove_qr_codes_from_pdf(text_removed)
    
    # Add cover page
    cover = Image.open("cover.png")
    final_images.insert(0, cover)
    
    # Save final PDF
    final_path = f"{file_path}_cleaned.pdf"
    final_images[0].save(final_path, format='PDF', save_all=True, 
                         append_images=final_images[1:])
    print(f"Cleaned PDF saved to: {final_path}")
    
    