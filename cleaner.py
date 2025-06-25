import numpy as np
import cv2

from PIL import Image
import pytesseract
from pyzbar.pyzbar import decode
from pdf2image import convert_from_path
import pandas as pd 


def preprocess_image(img_cv):
    """Enhances the image for better QR code detection, especially small ones."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)  # Adaptive Thresholding
    return enhanced

def enhance_qr(img_cv): 
    img_cv = cv2.resize(img_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_img = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(contrast_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 10)
    return binary



def replace_text_in_scanned_pdf(search_text, images):
    # Convert PDF pages to images
    modified_images = []

    count  = 1
    for img in images:
        # Convert to OpenCV format
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Extract text data
        h, w, _ = img_cv.shape
        img_width = img_cv.shape[1]  # Get page width

        d = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)

        for i, word in enumerate(d["text"]):
            if word.lower() == search_text:

                padding = 10 
                
                # Get text position
                x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
                
                x = max(x - padding, 0)
                y = max(y - padding, 0)

                w = w + 2 * padding
                h = h + 2 * padding

                # Estimate background color
                bg_patch = img_cv[max(y-5, 0):min(y+h+5, img_cv.shape[0]), max(x-5, 0):min(x+w+5, img_cv.shape[1])]
                avg_color = np.median(bg_patch, axis=(0, 1))  # Median background color

                # Cover the original text with the matching background color
                cv2.rectangle(img_cv, (x, y), (x + img_width, y + h), avg_color.tolist(), -1)
                # print(f"text in page number {count} has been removed")
        # Convert back to PIL format
        modified_images.append(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
        count += 1
    # Save modified images as PDF
    return modified_images


def remove_qr_codes_from_pdf(images):
    print("Removing QR codes from PDF...")
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


def pdf_processing(search_text: str, file_path: str): 
    pytesseract.pytesseract.tesseract_cmd = r"C:/Users/Sencho/anaconda3/Library/bin/tesseract.exe"
    print(f'converting {file_path} to images...')
    pdf_images = convert_from_path(f'{file_path}')
    print('converted pdf to images')

    editted_images = replace_text_in_scanned_pdf(search_text, pdf_images)
    final_images = remove_qr_codes_from_pdf(editted_images)
    
    cover = Image.open("cover.png")
    final_images.insert(0, cover)
    
    final_path = f"{file_path}_cleaned.pdf"
    final_images[0].save(final_path, format = 'PDF', save_all=True, append_images=final_images[1:])