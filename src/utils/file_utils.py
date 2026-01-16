import os
import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import PyPDF2
import pdfplumber
from docx import Document

from src.logging import get_logger
from src.exceptions import FileProcessingError

logger = get_logger(__name__)

# Allowed file formats
ALLOWED_FORMATS = ['pdf', 'docx', 'doc', 'txt']
DEFAULT_MAX_SIZE_MB = 10


def read_pdf(file_path: str) -> str:
    """
    Extract text from PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content
        
    Raises:
        FileProcessingError: If PDF reading fails
    """
    try:
        logger.info(f"Reading PDF file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        text_content = []
        
        # Try pdfplumber first (better for complex PDFs)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
            
            logger.info(f"Successfully extracted {len(text_content)} pages using pdfplumber")
            return '\n'.join(text_content)
            
        except Exception as plumber_error:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {plumber_error}")
            
            # Fallback to PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                
                logger.info(f"Successfully extracted {len(text_content)} pages using PyPDF2")
                return '\n'.join(text_content)
    
    except Exception as e:
        logger.error(f"Failed to read PDF: {file_path}")
        raise FileProcessingError(e, sys)


def read_docx(file_path: str) -> str:
    """
    Extract text from DOCX file.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        Extracted text content
        
    Raises:
        FileProcessingError: If DOCX reading fails
    """
    try:
        logger.info(f"Reading DOCX file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        doc = Document(file_path)
        
        # Extract text from paragraphs
        text_content = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text_content.append(cell.text)
        
        logger.info(f"Successfully extracted text from DOCX: {len(text_content)} sections")
        return '\n'.join(text_content)
    
    except Exception as e:
        logger.error(f"Failed to read DOCX: {file_path}")
        raise FileProcessingError(e, sys)


def read_txt(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Read text from plain text file.
    
    Args:
        file_path: Path to the text file
        encoding: File encoding (default: utf-8)
        
    Returns:
        File content as string
        
    Raises:
        FileProcessingError: If file reading fails
    """
    try:
        logger.info(f"Reading text file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try specified encoding first
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            logger.info(f"Successfully read text file with {encoding} encoding")
            return content
        
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding
            logger.warning(f"{encoding} encoding failed, trying latin-1")
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
            logger.info("Successfully read text file with latin-1 encoding")
            return content
    
    except Exception as e:
        logger.error(f"Failed to read text file: {file_path}")
        raise FileProcessingError(e, sys)


def validate_file_format(filename: str, allowed_formats: Optional[list] = None) -> bool:
    """
    Validate if file format is allowed.
    
    Args:
        filename: Name of the file
        allowed_formats: List of allowed extensions (default: ALLOWED_FORMATS)
        
    Returns:
        True if format is allowed, False otherwise
    """
    if allowed_formats is None:
        allowed_formats = ALLOWED_FORMATS
    
    file_extension = filename.split('.')[-1].lower()
    is_valid = file_extension in allowed_formats
    
    if is_valid:
        logger.debug(f"File format validated: {filename}")
    else:
        logger.warning(f"Invalid file format: {file_extension} (allowed: {allowed_formats})")
    
    return is_valid


def validate_file_size(file_path: str, max_size_mb: int = DEFAULT_MAX_SIZE_MB) -> bool:
    """
    Validate if file size is within limit.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum allowed size in megabytes
        
    Returns:
        True if size is valid, False otherwise
        
    Raises:
        FileProcessingError: If file doesn't exist
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        is_valid = file_size_mb <= max_size_mb
        
        if is_valid:
            logger.debug(f"File size validated: {file_size_mb:.2f}MB / {max_size_mb}MB")
        else:
            logger.warning(f"File too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB)")
        
        return is_valid
    
    except Exception as e:
        logger.error(f"Failed to validate file size: {file_path}")
        raise FileProcessingError(e, sys)


def save_uploaded_file(file_content: bytes, filename: str, upload_dir: str = "data/uploads") -> str:
    """
    Save uploaded file to directory with unique filename.
    
    Args:
        file_content: File content as bytes
        filename: Original filename
        upload_dir: Directory to save file (default: data/uploads)
        
    Returns:
        Path to saved file
        
    Raises:
        FileProcessingError: If save operation fails
    """
    try:
        # Create upload directory if it doesn't exist
        upload_path = Path(upload_dir)
        upload_path.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, extension = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{extension}"
        
        # Save file
        file_path = upload_path / unique_filename
        with open(file_path, 'wb') as file:
            file.write(file_content)
        
        logger.info(f"File saved successfully: {file_path}")
        return str(file_path)
    
    except Exception as e:
        logger.error(f"Failed to save file: {filename}")
        raise FileProcessingError(e, sys)


def get_file_metadata(file_path: str) -> Dict[str, any]:
    """
    Extract file metadata.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file metadata
        
    Raises:
        FileProcessingError: If metadata extraction fails
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat_info = os.stat(file_path)
        
        metadata = {
            'filename': os.path.basename(file_path),
            'extension': os.path.splitext(file_path)[1],
            'size_bytes': stat_info.st_size,
            'size_mb': stat_info.st_size / (1024 * 1024),
            'created_at': datetime.fromtimestamp(stat_info.st_ctime),
            'modified_at': datetime.fromtimestamp(stat_info.st_mtime),
            'absolute_path': os.path.abspath(file_path)
        }
        
        logger.debug(f"Extracted metadata for: {file_path}")
        return metadata
    
    except Exception as e:
        logger.error(f"Failed to get file metadata: {file_path}")
        raise FileProcessingError(e, sys)


def delete_file(file_path: str) -> bool:
    """
    Safely delete a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file was deleted, False if file didn't exist
        
    Raises:
        FileProcessingError: If deletion fails
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File doesn't exist, cannot delete: {file_path}")
            return False
        
        os.remove(file_path)
        logger.info(f"File deleted successfully: {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to delete file: {file_path}")
        raise FileProcessingError(e, sys)