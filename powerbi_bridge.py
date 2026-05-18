"""
Power BI Desktop model bridge.

Discovers any open Power BI Desktop instance via its local msmdsrv (Analysis
Services Tabular) endpoint and reads the model schema so the LLM can suggest
DAX/M against the student's actual table and column names.

Read-only. No write-back. Degrades gracefully if pyadomd is not installed
(empty endpoint list + None summary, helper still runs).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

ADOMD_DIR = r"C:\Program Files\Microsoft.NET\ADOMD.NET\160"


def _pyadomd():
    """Lazy-import pyadomd after registering the ADOMD.NET DLL path.

    Returns the Pyadomd class, or None if the dependency chain is missing.
    """
    if ADOMD_DIR not in sys.path and os.path.isdir(ADOMD_DIR):
        sys.path.append(ADOMD_DIR)
    try:
        import clr
        clr.AddReference("Microsoft.AnalysisServices.AdomdClient")
        from pyadomd import Pyadomd
        return Pyadomd
    except Exception:
        return None


def _read_port(port_file: Path) -> str | None:
    raw = port_file.read_bytes()
    for enc in ("utf-16-le", "utf-16", "utf-8-sig", "utf-8"):
        try:
            text = raw.decode(enc).strip().lstrip("﻿").strip()
        except UnicodeDecodeError:
            continue
        if text.isdigit():
            return text
    return None


def _workspace_ports() -> list[str]:
    """Read every Power BI Desktop workspace's msmdsrv.port.txt."""
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Power BI Desktop" / "AnalysisServicesWorkspaces"
    if not base.is_dir():
        return []
    ports: list[str] = []
    for ws in base.glob("AnalysisServicesWorkspace*"):
        pf = ws / "Data" / "msmdsrv.port.txt"
        if not pf.is_file():
            continue
        port = _read_port(pf)
        if port:
            ports.append(port)
    return ports


def _psutil_ports() -> list[str]:
    """Fallback: enumerate msmdsrv.exe TCP listeners on loopback."""
    try:
        import psutil
    except Exception:
        return []
    ports: list[str] = []
    for p in psutil.process_iter(["name"]):
        if (p.info.get("name") or "").lower() != "msmdsrv.exe":
            continue
        try:
            for c in p.connections(kind="tcp"):
                if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.ip in ("127.0.0.1", "::1"):
                    ports.append(str(c.laddr.port))
        except Exception:
            continue
    return ports


def _first_catalog(server: str) -> str | None:
    Pyadomd = _pyadomd()
    if Pyadomd is None:
        return None
    conn = f"Provider=MSOLAP;Data Source={server};"
    try:
        with Pyadomd(conn) as c:
            with c.cursor().execute("SELECT [CATALOG_NAME] FROM $SYSTEM.DBSCHEMA_CATALOGS") as cur:
                row = cur.fetchone()
                if row:
                    return str(row[0])
    except Exception:
        return None
    return None


def discover_pbi_endpoints() -> list[tuple[str, str]]:
    """Return [(server, database), ...] for every open Power BI Desktop instance."""
    seen: set[str] = set()
    endpoints: list[tuple[str, str]] = []
    for port in (*_workspace_ports(), *_psutil_ports()):
        if port in seen:
            continue
        seen.add(port)
        server = f"localhost:{port}"
        db = _first_catalog(server)
        if db:
            endpoints.append((server, db))
    return endpoints


def _query(server: str, database: str, dax: str) -> list[dict]:
    Pyadomd = _pyadomd()
    if Pyadomd is None:
        return []
    conn = f"Provider=MSOLAP;Data Source={server};Initial Catalog={database};"
    rows: list[dict] = []
    try:
        with Pyadomd(conn) as c:
            with c.cursor().execute(dax) as cur:
                cols = [d[0] for d in cur.description]
                for r in cur.fetchall():
                    rows.append(dict(zip(cols, r)))
    except Exception:
        return []
    return rows


# Tabular data-type ID -> friendly label.
# https://learn.microsoft.com/analysis-services/tmsl/columns-object-tmsl
_DTYPE = {
    1: "auto", 2: "string", 6: "int64", 8: "double", 9: "datetime",
    10: "decimal", 11: "boolean", 17: "binary", 19: "unknown", 20: "variant",
}

_CARDINALITY = {1: "one", 2: "many", 3: "none"}
_CROSSFILTER = {1: "OneDirection", 2: "BothDirections", 3: "Automatic"}


def read_model_summary(server: str, database: str) -> dict:
    """Return {tables, columns, relationships, measures} for one Power BI model."""
    tables_raw = _query(server, database, "SELECT * FROM $SYSTEM.TMSCHEMA_TABLES")
    columns_raw = _query(server, database, "SELECT * FROM $SYSTEM.TMSCHEMA_COLUMNS")
    rels_raw = _query(server, database, "SELECT * FROM $SYSTEM.TMSCHEMA_RELATIONSHIPS")
    measures_raw = _query(server, database, "SELECT * FROM $SYSTEM.TMSCHEMA_MEASURES")

    table_by_id = {
        t["ID"]: t.get("Name") for t in tables_raw if not t.get("IsHidden")
    }
    column_by_id = {c["ID"]: c for c in columns_raw}

    tables = []
    for t in tables_raw:
        if t.get("IsHidden"):
            continue
        cols = [
            {"name": c.get("ExplicitName") or c.get("Name"),
             "type": _DTYPE.get(c.get("ExplicitDataType") or c.get("DataType"), "unknown")}
            for c in columns_raw
            if c.get("TableID") == t["ID"]
            and not c.get("IsHidden")
            and (c.get("Type") or 1) == 1  # 1 = regular data column (exclude row-number/calc columns surrogates)
        ]
        tables.append({"name": t.get("Name"), "columns": cols})

    relationships = []
    for r in rels_raw:
        from_t = table_by_id.get(r.get("FromTableID"))
        to_t = table_by_id.get(r.get("ToTableID"))
        from_c = column_by_id.get(r.get("FromColumnID"), {}).get("ExplicitName")
        to_c = column_by_id.get(r.get("ToColumnID"), {}).get("ExplicitName")
        if not (from_t and to_t and from_c and to_c):
            continue
        relationships.append({
            "from": f"{from_t}[{from_c}]",
            "to": f"{to_t}[{to_c}]",
            "from_cardinality": _CARDINALITY.get(r.get("FromCardinality"), "?"),
            "to_cardinality": _CARDINALITY.get(r.get("ToCardinality"), "?"),
            "cross_filter": _CROSSFILTER.get(r.get("CrossFilteringBehavior"), "?"),
            "active": bool(r.get("IsActive", True)),
        })

    measures = []
    for m in measures_raw:
        table = table_by_id.get(m.get("TableID"))
        if not table:
            continue
        measures.append({
            "table": table,
            "name": m.get("Name"),
            "expression": (m.get("Expression") or "").strip(),
        })

    return {
        "server": server,
        "database": database,
        "tables": tables,
        "relationships": relationships,
        "measures": measures,
    }


def format_markdown(summary: dict) -> str:
    """Render the summary as a compact markdown block for the LLM context."""
    if not summary or not summary.get("tables"):
        return ""
    lines: list[str] = ["### Live Power BI model"]
    for t in summary["tables"]:
        cols = ", ".join(f"{c['name']}:{c['type']}" for c in t["columns"])
        lines.append(f"- `{t['name']}` — {cols}")
    rels = summary.get("relationships") or []
    if rels:
        lines.append("")
        lines.append("**Relationships**")
        for r in rels:
            arrow = " <->" if r["cross_filter"] == "BothDirections" else " ->"
            mark = "" if r["active"] else "  (inactive)"
            lines.append(f"- {r['from']}{arrow} {r['to']}  [{r['from_cardinality']}:{r['to_cardinality']}]{mark}")
    meas = summary.get("measures") or []
    if meas:
        lines.append("")
        lines.append("**Existing measures**")
        for m in meas:
            expr = m["expression"].replace("\n", " ").strip()
            if len(expr) > 160:
                expr = expr[:157] + "..."
            lines.append(f"- `{m['table']}[{m['name']}]` = {expr}")
    return "\n".join(lines)


def build_context_block() -> tuple[str, list[dict]]:
    """One-shot helper used by the launcher. Returns (markdown, raw_summaries)."""
    endpoints = discover_pbi_endpoints()
    summaries = [read_model_summary(s, d) for s, d in endpoints]
    parts = [format_markdown(s) for s in summaries if s.get("tables")]
    markdown = "\n\n".join(p for p in parts if p)
    return markdown, summaries


if __name__ == "__main__":
    eps = discover_pbi_endpoints()
    print(f"Endpoints: {eps}")
    md, raw = build_context_block()
    print()
    print(md or "(no model context)")
