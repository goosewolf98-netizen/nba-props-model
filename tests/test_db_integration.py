import unittest
import sys
import os
import pandas as pd
import sqlite3
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from db.manager import DBManager

class TestDBManager(unittest.TestCase):
    def setUp(self):
        # Use a temporary DB
        self.test_db_path = Path("tests/test_nba.db")
        if self.test_db_path.exists():
            self.test_db_path.unlink()
        self.db = DBManager(db_path=self.test_db_path)

    def tearDown(self):
        if self.test_db_path.exists():
            self.test_db_path.unlink()

    def test_init_db(self):
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall()]
        self.assertIn('boxscores', tables)
        self.assertIn('prop_lines', tables)
        self.assertIn('reddit_sentiment', tables)
        conn.close()

    def test_upsert_boxscores(self):
        df = pd.DataFrame({
            'GAME_ID': ['1', '1'],
            'PLAYER_ID': [101, 102],
            'PTS': [20, 15]
        })
        self.db.upsert_boxscores(df)

        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT player_id, pts FROM boxscores ORDER BY player_id")
        rows = cursor.fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], (101, 20))
        self.assertEqual(rows[1], (102, 15))
        conn.close()

    def test_upsert_prop_lines(self):
        df = pd.DataFrame({
            'game_date': ['2023-01-01'],
            'player': ['Player A'],
            'stat': ['pts'],
            'line': [20.5],
            'book': ['Book A']
        })
        self.db.upsert_prop_lines(df)

        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT line FROM prop_lines")
        row = cursor.fetchone()
        self.assertEqual(row[0], 20.5)
        conn.close()

if __name__ == '__main__':
    unittest.main()
