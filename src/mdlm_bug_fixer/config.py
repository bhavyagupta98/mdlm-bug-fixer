"""
Configuration module for MDLM Bug Fixer
"""


class Config:
    """
    Configuration class for MDLM Bug Fixer
    """
    # Default configuration values
    DEBUG = False
    VERSION = "0.1.0"
    
    @classmethod
    def get_config(cls):
        """
        Get current configuration
        """
        return {
            "debug": cls.DEBUG,
            "version": cls.VERSION,
        }
