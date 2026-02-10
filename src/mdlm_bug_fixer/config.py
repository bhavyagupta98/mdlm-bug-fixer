"""
Configuration module for MDLM Bug Fixer
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.1.0"


class Config:
    """
    Configuration class for MDLM Bug Fixer
    """
    # Default configuration values
    DEBUG = False
    VERSION = __version__
    
    @classmethod
    def get_config(cls):
        """
        Get current configuration
        """
        return {
            "debug": cls.DEBUG,
            "version": cls.VERSION,
        }
