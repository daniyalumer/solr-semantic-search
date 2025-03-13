import os
import re
import pandas as pd
import unicodedata
from PyPDF2 import PdfReader
from docx import Document
from typing import List, Dict
from pathlib import Path
import subprocess
import magic

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_text_from_doc(doc_path: str) -> str:
    """Extract text content from a DOC file using catdoc."""
    try:
        result = subprocess.run(['catdoc', doc_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"Error extracting text from {doc_path}: {result.stderr}")
            return ""
    except Exception as e:
        print(f"Error extracting text from {doc_path}: {e}")
        return ""

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text content from a DOCX file."""
    try:
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {docx_path}: {e}")
        return ""

def sanitize_text(text: str) -> str:
    """Sanitize input text to remove invalid control characters while preserving Unicode."""
    # Remove all ASCII control characters except tabs (\t), newlines (\n, \r)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u0080-\uFFFF]', '', text)
    
    # Normalize Unicode (NFKC) to standardize text representation
    text = unicodedata.normalize("NFKC", text)
    
    return text.strip()
    
def preprocess_text(text: str) -> str:
    """Preprocess the input text by sanitizing it."""
    sanitized_text = sanitize_text(text)
    return sanitized_text

def process_files(base_dir: str, output_csv: str, file_types: List[str], filter_set: set = None) -> pd.DataFrame:
    """Recursively process all specified file types in directory and subdirectories."""
    data = []
    base_path = Path(base_dir)
    mime = magic.Magic(mime=True)
    skipped_files = 0

    # Walk through all subdirectories
    for root, _, files in os.walk(base_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_extension = filename.lower().split('.')[-1]
            mime_type = mime.from_file(file_path)
            base_filename = filename.rsplit('-', 1)[0]

            if file_extension in file_types and (filter_set is None or base_filename in filter_set):
                if mime_type == 'application/pdf':
                    extracted_text = extract_text_from_pdf(file_path)
                elif mime_type == 'application/msword':
                    extracted_text = extract_text_from_doc(file_path)
                elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    extracted_text = extract_text_from_docx(file_path)
                else:
                    print(f"Unsupported file type: {mime_type}")
                    continue

                # Preprocess the extracted text
                preprocessed_text = preprocess_text(extracted_text)

                # Get relative path for better organization
                relative_path = os.path.relpath(file_path, base_path)
                
                # Add to data list
                data.append({
                    "filename": filename,
                    "path": relative_path,
                    "raw_text": extracted_text,
                    "preprocessed_text": preprocessed_text 
                })
                print(f"Processed: {relative_path}")
            else:
                skipped_files += 1

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort DataFrame by filename
    df = df.sort_values(by="filename")
    
    # Save DataFrame to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"Skipped {skipped_files} files from PROFILE because there was no corresponding file in CV.")
    
    return df

def get_base_filenames(directory: str, suffix: str) -> set:
    """Get a set of base filenames (without extensions) from a directory."""
    base_filenames = set()
    for root, _, files in os.walk(directory):
        for filename in files:
            if suffix in filename:
                base_filename = filename.rsplit('-', 1)[0]
                base_filenames.add(base_filename)
    return base_filenames