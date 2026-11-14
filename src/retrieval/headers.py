"""Readable passage headers from each document's source path.

Paths carry context the title often lacks, for example
`slack/devex/1759999992-opaque-routing-shim-sdk-discuss.json` (a Slack title is only the
channel name). The header turns that into
`Slack #devex · 2025-10-09 · opaque routing shim sdk discuss`: the platform, the folders
(channel, mailbox, project, team), a date when the file name starts with one, and the file
name's words (topic, ticket key, customer). It uses only the corpus's own paths, never labels.
About 8% of documents are deliberately misfiled in this dataset, so folders can be wrong.
"""
import re
from datetime import datetime, timezone

HEADER_VERSION = "path-v1"
SOURCES = {"slack": "Slack", "gmail": "Gmail", "linear": "Linear", "jira": "Jira",
           "github": "GitHub", "confluence": "Confluence", "google_drive": "Google Drive",
           "hubspot": "HubSpot", "fireflies": "Fireflies"}
GENERIC_FOLDERS = {"shared_drives", "my_drive"}


def _date(stem: str) -> tuple[str | None, str]:
    """A leading date in the file name, as YYYY-MM-DD, and the rest of the name."""
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})[-_](.*)", stem)
    if match:
        return f"{match[1]}-{match[2]}-{match[3]}", match[4]
    match = re.match(r"(\d{4})(\d{2})(\d{2})[-_](.*)", stem)
    if match and 1 <= int(match[2]) <= 12 and 1 <= int(match[3]) <= 31:
        return f"{match[1]}-{match[2]}-{match[3]}", match[4]
    match = re.match(r"(\d{10})[-_](.*)", stem)
    if match:  # Unix seconds (Slack thread files).
        moment = datetime.fromtimestamp(int(match[1]), tz=timezone.utc)
        if 2015 <= moment.year <= 2035:
            return moment.strftime("%Y-%m-%d"), match[2]
    return None, stem


def path_header(source_type: str, path: str | None) -> str:
    """One line of context for a document; empty when there is no path."""
    if not path:
        return ""
    parts = path.removesuffix(".json").split("/")
    source = SOURCES.get(parts[0], source_type or parts[0])
    folders = [p.replace("_", " ") for p in parts[1:-1] if p not in GENERIC_FOLDERS]
    if parts[0] == "slack" and folders:
        folders[0] = "#" + folders[0]
    date, rest = _date(parts[-1]) if len(parts) > 1 else (None, "")
    key = re.match(r"([A-Z][A-Z0-9]*-\d+)[-_](.*)", rest)  # Ticket keys stay intact.
    if key:
        rest = key[2]
    topic = re.sub(r"[-_]+", " ", rest).strip()
    pieces = [source, " / ".join(folders), date, key[1] if key else None, topic]
    return " · ".join(p for p in pieces if p)
