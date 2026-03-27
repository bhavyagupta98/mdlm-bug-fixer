"""
Tests for utility functions
"""
import unittest
from src.mdlm_bug_fixer.utils import validate_input, format_output


class TestUtils(unittest.TestCase):
    """
    Test cases for utility functions
    """
    
    def test_validate_input_with_none(self):
        """
        Test validate_input with None
        """
        self.assertFalse(validate_input(None))
    
    def test_validate_input_with_data(self):
        """
        Test validate_input with valid data
        """
        self.assertTrue(validate_input("test"))
        self.assertTrue(validate_input(123))
        self.assertTrue(validate_input({"key": "value"}))
    
    def test_format_output(self):
        """
        Test format_output function
        """
        self.assertEqual(format_output("test"), "test")
        self.assertEqual(format_output(123), "123")


if __name__ == "__main__":
    unittest.main()
