from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import re
import subprocess
import sys
from urllib.request import urlopen, Request

import pandas as pd

ART_DIR = Path("artifacts")

TEAM_ABBRS = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
}


def http_get_text(url: str, timeout=20) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


def http_get_bytes(url: str, timeout=30) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def ensure_pypdf():
    if importlib.util.find_spec("pypdf") is not None:
        return True, None
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pypdf"])
    except Exception as e:
        return False, str(e)
    if importlib.util.find_spec("pypdf") is None:
        return False, "pypdf installation failed"
    return True, None


def find_latest_injury_pdf():
    season_page = "https://official.nba.com/nba-injury-report-2025-26-season/"
    try:
        html = http_get_text(season_page)
        urls = re.findall(r"https?://[^\"'\s]+Injury-Report_\d{4}-\d{2}-\d{2}_[^\"'\s]+\.pdf", html)
        if not urls:
            rels = re.findall(r"/referee/injury/Injury-Report_\d{4}-\d{2}-\d{2}_[^\"'\s]+\.pdf", html)
            urls = ["https://ak-static.cms.nba.com" + r for r in rels]

        def key(u):
            m = re.search(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})(AM|PM)\.pdf", u)
            if not m:
                return ("0000-00-00", 0, 0)
            d, hh, mm, ap = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            h24 = hh % 12 + (12 if ap == "PM" else 0)
            return (d, h24, mm)

        urls = sorted(set(urls), key=key)
        return urls[-1] if urls else None
    except Exception:
        return None


def parse_report_datetime_from_url(url: str) -> str | None:
    m = re.search(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})(AM|PM)\.pdf", url or "")
    if not m:
        m_date = re.search(r"Injury-Report_(\d{4}-\d{2}-\d{2})_", url or "")
        return m_date.group(1) if m_date else None
    date_str, hh, mm, ap = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    h24 = hh % 12 + (12 if ap == "PM" else 0)
    return f"{date_str} {h24:02d}:{mm:02d}:00"


def parse_injury_report(text: str) -> pd.DataFrame:
    rows = []
    status_words = ["OUT", "QUESTIONABLE", "PROBABLE", "DOUBTFUL", "AVAILABLE"]
    status_pattern = "|".join(status_words)
    line_re = re.compile(
        rf"^(?P<team>[A-Z]{{3}})\s+(?P<player>[A-Za-z\.'\-]+(?:\s+[A-Za-z\.'\-]+){{0,3}})\s+"
        rf"(?P<status>{status_pattern})\b(?P<reason>.*)$",
        re.IGNORECASE,
    )
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw.strip())
        if not line:
            continue
        m = line_re.match(line)
        if not m:
            continue
        team = m.group("team").upper()
        if team not in TEAM_ABBRS:
            continue
        rows.append({
            "team_abbr": team,
            "player": m.group("player").strip(),
            "status": m.group("status").upper(),
            "reason": m.group("reason").strip(),
        })
    return pd.DataFrame(rows)


def fetch_latest_injuries(out_path: Path) -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    pdf_url = find_latest_injury_pdf()
    if not pdf_url:
        pd.DataFrame(columns=["report_datetime", "team_abbr", "player", "status", "reason"]).to_csv(out_path, index=False)
        print("Could not find latest official NBA injury PDF.")
        return

    ok, err = ensure_pypdf()
    if not ok:
        pd.DataFrame(columns=["report_datetime", "team_abbr", "player", "status", "reason"]).to_csv(out_path, index=False)
        print(f"Could not install/read pypdf: {err}")
        return

    try:
        from pypdf import PdfReader
        import io
        pdf_bytes = http_get_bytes(pdf_url)
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join([(p.extract_text() or "") for p in reader.pages])
        df = parse_injury_report(text)
        report_datetime = parse_report_datetime_from_url(pdf_url)
        df["report_datetime"] = report_datetime
        df = df[["report_datetime", "team_abbr", "player", "status", "reason"]]
        df.to_csv(out_path, index=False)
        print("Saved injuries to", out_path)
    except Exception as e:
        pd.DataFrame(columns=["report_datetime", "team_abbr", "player", "status", "reason"]).to_csv(out_path, index=False)
        print(f"Official injury report parse failed safely: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fetch official NBA injury report PDF.")
    parser.add_argument("--out", default=str(ART_DIR / "injuries_latest.csv"))
    args = parser.parse_args()
    fetch_latest_injuries(Path(args.out))


if __name__ == "__main__":
    main()
