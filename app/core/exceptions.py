class DocuMindException(Exception):
    """Base exception for the application."""
    pass

class UnsupportedFileTypeError(DocuMindException):
    """Raised when an uploaded file type is not supported."""
    pass

class FileTooLargeError(DocuMindException):
    """Raised when an uploaded file exceeds the max size limit."""
    pass

class DocumentIngestionError(DocuMindException):
    """Raised when document ingestion fails."""
    pass
