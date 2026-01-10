from pathlib import Path
import requests
import pandas as pd

OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
INJ_URL = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/teams/{team_id}/injuries?lang=en&region=us"

HEADERS = {"User-Agent": "Mozilla/5.0 (nba-props-model/1.0)"}
CACHE = {}

def write_error(msg: str):
    (OUT_DIR / "espn_injuries_error.txt").write_text(msg)
    (OUT_DIR / "espn_injuries.csv").write_text("error\n" + msg + "\n")

def get_json(url: str):
    if url in CACHE:
        return CACHE[url]
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    j = r.json()
    CACHE[url] = j
    return j

def resolve(obj):
    # ESPN core API often returns {"$ref": "..."} objects
    if isinstance(obj, dict) and "$ref" in obj and len(obj.keys()) == 1:
        return get_json(obj["$ref"])
    return obj

def extract_team_ids(teams_json: dict):
    out = []
    leagues = teams_json.get("sports", [{}])[0].get("leagues", [{}])[0]
    for entry in leagues.get("teams", []):
        team = entry.get("team", {})
        tid = str(team.get("id", "")).strip()
        name = team.get("displayName") or team.get("name") or tid
        if tid:
            out.append((tid, name))
    return out

def to_row(team_id, team_name, injury_obj):
    injury_obj = resolve(injury_obj)

    athlete = resolve(injury_obj.get("athlete", {}))
    status = resolve(injury_obj.get("status", {}))

    player = athlete.get("displayName") or athlete.get("fullName") or ""
    status_type = ""
    if isinstance(status, dict):
        status_type = status.get("type") or status.get("name") or ""
    else:
        status_type = str(status)

    detail = injury_obj.get("details") or injury_obj.get("comment") or injury_obj.get("description") or ""
    date = injury_obj.get("date") or injury_obj.get("updated") or ""
    return_date = injury_obj.get("estimatedReturnDate") or injury_obj.get("returnDate") or ""

    return {
        "team": team_name,
        "team_id": team_id,
        "player": player,
        "status": status_type,
        "detail": detail,
        "date": date,
        "estimated_return_date": return_date,
    }

def main():
    try:
        teams_json = get_json(TEAMS_URL)
        teams = extract_team_ids(teams_json)
        if not teams:
            write_error("No teams found from ESPN teams endpoint.")
            return

        rows = []
        for team_id, team_name in teams:
            try:
                inj_json = get_json(INJ_URL.format(team_id=team_id))
                inj_json = resolve(inj_json)

                items = inj_json.get("items") or inj_json.get("injuries") or []
                if isinstance(items, dict):
                    items = [items]

                for it in items:
                    # follow $ref to get the real injury object
                    resolved = resolve(it)
                    # skip if still not a dict
                    if not isinstance(resolved, dict):
                        continue
                    rows.append(to_row(team_id, team_name, resolved))

            except Exception:
                continue

        df = pd.DataFrame(rows)
        out_csv = OUT_DIR / "espn_injuries.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved {out_csv} rows={len(df)} cols={len(df.columns)}")

        if len(df) == 0:
            write_error("Injuries endpoint returned 0 rows after expanding refs (ESPN format may have changed).")

    except Exception as e:
        write_error(f"ESPN injuries fetch failed: {e}")

if __name__ == "__main__":
    main()
