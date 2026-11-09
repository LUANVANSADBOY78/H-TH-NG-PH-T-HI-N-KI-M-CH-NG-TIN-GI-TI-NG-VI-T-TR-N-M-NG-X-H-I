"""
Unit tests for Auth and Permission modules.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.models import init_tables
from src.auth.login import register_user, authenticate_user
from src.auth.permissions import can_access_admin_panel, can_manage_users

class TestAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_tables()

    def test_admin_authentication(self):
        ok, user, msg = authenticate_user("admin", "admin123")
        self.assertTrue(ok)
        self.assertEqual(user["role"], "admin")
        self.assertTrue(can_access_admin_panel(user["role"]))
        self.assertTrue(can_manage_users(user["role"]))

    def test_invalid_login(self):
        ok, user, msg = authenticate_user("admin", "wrongpassword")
        self.assertFalse(ok)

if __name__ == "__main__":
    unittest.main()

