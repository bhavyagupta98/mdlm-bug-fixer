"""
Utility functions for MDLM Bug Fixer
"""


def validate_input(data):
    """
    Validate input data
    
    Args:
        data: Input data to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if data is None:
        return False
    return True


def format_output(result):
    """
    Format output for display
    
    Args:
        result: Result to format
        
    Returns:
        str: Formatted result
    """
    return str(result)
