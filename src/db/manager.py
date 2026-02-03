import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/nba.db")

class DBManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        # Boxscores Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS boxscores (
                game_id TEXT,
                player_id INTEGER,
                team_id INTEGER,
                game_date TEXT,
                player_name TEXT,
                team_abbreviation TEXT,
                pts INTEGER,
                reb INTEGER,
                ast INTEGER,
                stl INTEGER,
                blk INTEGER,
                tpm INTEGER,
                min TEXT,
                fga INTEGER,
                fgm INTEGER,
                fg3a INTEGER,
                fg3m INTEGER,
                fta INTEGER,
                ftm INTEGER,
                plus_minus INTEGER,
                PRIMARY KEY (game_id, player_id)
            )
        """)

        # Team Boxscores Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_boxscores (
                game_id TEXT,
                team_id INTEGER,
                game_date TEXT,
                team_abbreviation TEXT,
                pts INTEGER,
                pts_paint INTEGER,
                pts_2nd_chance INTEGER,
                pts_fb INTEGER,
                largest_lead INTEGER,
                lead_changes INTEGER,
                times_tied INTEGER,
                fga INTEGER,
                fgm INTEGER,
                fg3a INTEGER,
                fg3m INTEGER,
                fta INTEGER,
                ftm INTEGER,
                reb INTEGER,
                oreb INTEGER,
                dreb INTEGER,
                ast INTEGER,
                stl INTEGER,
                blk INTEGER,
                tov INTEGER,
                pf INTEGER,
                plus_minus INTEGER,
                min TEXT,
                PRIMARY KEY (game_id, team_id)
            )
        """)

        # Prop Lines Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prop_lines (
                game_date TEXT,
                player TEXT,
                stat TEXT,
                line REAL,
                over_odds REAL,
                under_odds REAL,
                book TEXT,
                PRIMARY KEY (game_date, player, stat, book)
            )
        """)

        # Reddit Sentiment Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reddit_sentiment (
                id TEXT PRIMARY KEY,
                subreddit TEXT,
                title TEXT,
                text TEXT,
                score INTEGER,
                num_comments INTEGER,
                url TEXT,
                created_utc REAL,
                scraped_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    def upsert_boxscores(self, df: pd.DataFrame):
        """
        Upserts player boxscores.
        Expects DataFrame cols matching table cols (case-insensitive mapping handled mainly by caller,
        but we'll do some basic cleanup).
        """
        if df.empty:
            return

        conn = self.get_connection()

        # Standardize columns to match DB schema if possible
        # This is a basic mapping based on typical NBA API returns
        col_map = {
            'GAME_ID': 'game_id',
            'PLAYER_ID': 'player_id',
            'TEAM_ID': 'team_id',
            'GAME_DATE': 'game_date',
            'PLAYER_NAME': 'player_name',
            'TEAM_ABBREVIATION': 'team_abbreviation',
            'PTS': 'pts',
            'REB': 'reb',
            'AST': 'ast',
            'STL': 'stl',
            'BLK': 'blk',
            'FG3M': 'tpm', # tpm in DB, FG3M in API
            'MIN': 'min',
            'FGA': 'fga',
            'FGM': 'fgm',
            'FG3A': 'fg3a',
            # 'FG3M' is already mapped to tpm, but we might want it as fg3m too if schema had it.
            # Schema has fg3m and tpm? No, just tpm (which is fg3m).
            # Wait, schema has `tpm` AND `fg3m`. Let's just map FG3M to `fg3m` and `tpm`.
            'FTA': 'fta',
            'FTM': 'ftm',
            'PLUS_MINUS': 'plus_minus'
        }

        # Lowercase columns for easier mapping
        df.columns = [c.upper() for c in df.columns]

        # Rename available columns
        rename_dict = {k: v for k, v in col_map.items() if k in df.columns}
        df_to_load = df.rename(columns=rename_dict)

        # Ensure we have the mapped columns, others can be ignored or handled
        # We need to filter to only columns that exist in the table
        valid_cols = [
            'game_id', 'player_id', 'team_id', 'game_date', 'player_name',
            'team_abbreviation', 'pts', 'reb', 'ast', 'stl', 'blk', 'tpm',
            'min', 'fga', 'fgm', 'fg3a', 'fg3m', 'fta', 'ftm', 'plus_minus'
        ]

        # Handle 'tpm' alias if FG3M was mapped to it?
        # Actually in the dict above I mapped FG3M -> tpm.
        # But let's check the schema again.
        # I defined: tpm INTEGER, fg3m INTEGER.
        # Ideally tpm IS fg3m. I should probably just populate both or pick one.
        # Let's map FG3M to fg3m, and also fill tpm with fg3m if tpm is missing.

        if 'FG3M' in df.columns:
            df_to_load['fg3m'] = df['FG3M']
            df_to_load['tpm'] = df['FG3M']

        # Keep only valid columns
        df_final = df_to_load[[c for c in valid_cols if c in df_to_load.columns]]

        # SQLite upsert is tricky with pandas.to_sql.
        # We will iterate and execute INSERT OR REPLACE for safety/simplicity
        # given the volume isn't massive (thousands of rows, not millions per run).
        # Or use a temporary table.

        # Temp table approach is faster
        df_final.to_sql('temp_boxscores', conn, if_exists='replace', index=False)

        cols = ', '.join(df_final.columns)
        placeholders = ', '.join(['?'] * len(df_final.columns)) # Not used in bulk select, but good for reference

        # Build the dynamic SQL
        # We need to know which columns we actually have to build the INSERT statement
        actual_cols = df_final.columns.tolist()
        cols_str = ', '.join(actual_cols)

        query = f"""
            INSERT OR REPLACE INTO boxscores ({cols_str})
            SELECT {cols_str} FROM temp_boxscores
        """

        cursor = conn.cursor()
        cursor.execute(query)
        cursor.execute("DROP TABLE temp_boxscores")

        conn.commit()
        conn.close()
        print(f"Upserted {len(df)} boxscore rows.")

    def upsert_team_boxscores(self, df: pd.DataFrame):
        if df.empty:
            return

        conn = self.get_connection()
        df.columns = [c.upper() for c in df.columns]

        # Basic mapping
        # API often gives: TEAM_ID, TEAM_ABBREVIATION, GAME_ID, GAME_DATE, MIN, PTS, FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PLUS_MINUS
        col_map = {
            'GAME_ID': 'game_id',
            'TEAM_ID': 'team_id',
            'GAME_DATE': 'game_date',
            'TEAM_ABBREVIATION': 'team_abbreviation',
            'PTS': 'pts',
            'FGA': 'fga', 'FGM': 'fgm',
            'FG3A': 'fg3a', 'FG3M': 'fg3m',
            'FTA': 'fta', 'FTM': 'ftm',
            'REB': 'reb', 'OREB': 'oreb', 'DREB': 'dreb',
            'AST': 'ast', 'STL': 'stl', 'BLK': 'blk',
            'TOV': 'tov', 'PF': 'pf',
            'PLUS_MINUS': 'plus_minus',
            'MIN': 'min'
        }

        rename_dict = {k: v for k, v in col_map.items() if k in df.columns}
        df_to_load = df.rename(columns=rename_dict)

        valid_cols = [
            'game_id', 'team_id', 'game_date', 'team_abbreviation',
            'pts', 'fga', 'fgm', 'fg3a', 'fg3m', 'fta', 'ftm',
            'reb', 'oreb', 'dreb', 'ast', 'stl', 'blk', 'tov', 'pf', 'plus_minus', 'min'
        ]

        df_final = df_to_load[[c for c in valid_cols if c in df_to_load.columns]]

        df_final.to_sql('temp_team_box', conn, if_exists='replace', index=False)

        actual_cols = df_final.columns.tolist()
        cols_str = ', '.join(actual_cols)

        query = f"""
            INSERT OR REPLACE INTO team_boxscores ({cols_str})
            SELECT {cols_str} FROM temp_team_box
        """

        cursor = conn.cursor()
        cursor.execute(query)
        cursor.execute("DROP TABLE temp_team_box")

        conn.commit()
        conn.close()
        print(f"Upserted {len(df)} team boxscore rows.")

    def upsert_prop_lines(self, df: pd.DataFrame):
        if df.empty:
            return

        conn = self.get_connection()

        # Ensure columns exist
        required_cols = ['game_date', 'player', 'stat', 'line', 'book']
        for col in required_cols:
            if col not in df.columns:
                print(f"Missing required column {col} for prop_lines upsert.")
                return

        # Optional columns fill with None if missing
        optional = ['over_odds', 'under_odds']
        for col in optional:
            if col not in df.columns:
                df[col] = None

        df_final = df[required_cols + optional]

        df_final.to_sql('temp_props', conn, if_exists='replace', index=False)

        cols_str = ', '.join(df_final.columns)

        query = f"""
            INSERT OR REPLACE INTO prop_lines ({cols_str})
            SELECT {cols_str} FROM temp_props
        """

        cursor = conn.cursor()
        cursor.execute(query)
        cursor.execute("DROP TABLE temp_props")

        conn.commit()
        conn.close()
        print(f"Upserted {len(df)} prop lines.")

    def insert_sentiment(self, posts: list):
        if not posts:
            return

        conn = self.get_connection()
        cursor = conn.cursor()

        query = """
            INSERT OR IGNORE INTO reddit_sentiment
            (id, subreddit, title, text, score, num_comments, url, created_utc, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        data = []
        now = datetime.now().isoformat()
        for p in posts:
            data.append((
                p.get('id'),
                p.get('subreddit'),
                p.get('title'),
                p.get('text', '')[:5000], # Truncate large text
                p.get('score'),
                p.get('num_comments'),
                p.get('url'),
                p.get('created_utc'),
                now
            ))

        cursor.executemany(query, data)
        conn.commit()
        conn.close()
        print(f"Inserted {len(data)} reddit posts.")

if __name__ == "__main__":
    db = DBManager()
    print(f"Database initialized at {db.db_path}")
