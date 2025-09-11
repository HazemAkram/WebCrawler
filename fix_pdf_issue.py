import os
import shutil

# Path to the root output directory
OUTPUT_DIR = "output"

# Path to your new fixed PDF
NEW_PDF = "omegamotor_catalog_en.pdf"

# Walk through the output directory
for root, dirs, files in os.walk(OUTPUT_DIR):
    for file in files:
        if file.lower() == " omegamotor_catalog_en.pdf":
            old_path = os.path.join(root, file)
            print(f"Replacing: {old_path}")
            try:
                shutil.copy(NEW_PDF, old_path)  # overwrite with the new PDF
            except Exception as e:
                print(f"Error replacing {old_path}: {e}")
