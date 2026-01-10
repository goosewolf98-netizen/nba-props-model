from pathlib import Path
import json
import requests
import pandas as pd

OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
CORE_INJ_URL = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/teams/{team_id}/injuries?lang=en&region=us"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; nba-props-model/1.0)"
}

def write_error(msg: str):
    (OUT_DIR / "espn_injuries_error.txt").write_text(msg)
    (OUT_DIR / "espn_injuries.csv").write_text("error\n" + msg + "\n")

def safe_get(url: str):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def extract_team_ids(teams_json: dict) -> list[tuple[str, str]]:
    # Returns list of (team_id, team_name)
    out = []
    for entry in teams_json.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
        team = entry.get("team", {})
        tid = str(team.get("id", "")).strip()
        name = team.get("displayName") or team.get("name") or tid
        if tid:
            out.append((tid, name))
    return out

def flatten_injuries(team_id: str, team_name: str, inj_json: dict) -> list[dict]:
    rows = []

    # Common patterns: "items" list OR direct list fields
    items = inj_json.get("items") or inj_json.get("injuries") or []
    # Sometimes it's not expanded and only has "$ref" links
    if isinstance(items, dict):
        items = [items]

    for it in items:
        # If item is a ref object, skip (we'd need extra calls)
        if isinstance(it, dict) and "$ref" in it and len(it.keys()) == 1:
            continue

        athlete = it.get("athlete", {}) if isinstance(it, dict) else {}
        rows.append({
            "team": team_name,
            "team_id": team_id,
            "player": athlete.get("displayName") or athlete.get("fullName") or it.get("fullName") or "",
            "status": it.get("status", {}).get("type") if isinstance(it.get("status"), dict) else it.get("status"),
            "detail": it.get("details") or it.get("comment") or it.get("description") or "",
            "date": it.get("date") or it.get("updated") or "",
            "estimated_return_date": it.get("estimatedReturnDate") or it.get("returnDate") or "",
        })

    return rows

def main():
    try:
        teams_json = safe_get(TEAMS_URL)
        teams = extract_team_ids(teams_json)
        if not teams:
            write_error("No teams found from ESPN teams endpoint.")
            return

        all_rows = []
        for team_id, team_name in teams:
            try:
                inj_json = safe_get(CORE_INJ_URL.format(team_id=team_id))
                all_rows.extend(flatten_injuries(team_id, team_name, inj_json))
            except Exception:
                # don't fail the entire workflow if one team endpoint fails
                continue

        df = pd.DataFrame(all_rows)
        out_csv = OUT_DIR / "espn_injuries.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved {out_csv} rows={len(df)} cols={len(df.columns)}")

        if len(df) == 0:
            write_error("Fetched injuries but got 0 rows (endpoint may require expanding refs).")

    except Exception as e:
        write_error(f"ESPN injuries fetch failed: {e}")

if __name__ == "__main__":
    main()
