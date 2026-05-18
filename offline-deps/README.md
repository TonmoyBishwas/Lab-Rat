# Offline dependencies — Power BI helper

Pre-staged so `start_powerbi.py` can read the live Power BI Desktop model
without any internet during the exam.

## What's here

- `wheels/` — Python packages
  - `pythonnet`, `clr_loader`, `cffi`, `pycparser` — the .NET <-> Python bridge
  - `psutil` — fallback process scan for the msmdsrv port
  - `pyadomd` — DAX/MDX over the local Analysis Services endpoint (source dist, no compiled code)

## What you must install **once** on the lab PC

1. **Microsoft Analysis Services client libraries.**
   Download two MSIs from this page (do this on any internet machine, copy to the lab PC):
   https://learn.microsoft.com/analysis-services/client-libraries

   - `SQL_AS_AMO.msi` (Analysis Management Objects)
   - `SQL_AS_ADOMD.msi` (ADOMD.NET)

   Both install silently. They land DLLs under
   `C:\Program Files\Microsoft.NET\ADOMD.NET\160\` — the path `powerbi_bridge.py`
   appends to `sys.path` at import time.

2. **Python wheels.** From this directory:
   ```powershell
   python -m pip install --no-index --find-links offline-deps\wheels `
       pythonnet psutil pyadomd
   ```
   (.NET Framework 4.7.2+ is already on Windows 11 — no separate install.)

## How to verify

With one of the lab `.xlsx` files open in **Power BI Desktop**:

```powershell
python powerbi_bridge.py
```

Expected output: one or more `(localhost:NNNNN, GUID)` endpoint tuples plus
a markdown block listing the tables, columns, relationships, and any
measures the student has created.

If you see `Endpoints: []` with no markdown:
- Confirm a `.pbix` is actually open in Power BI Desktop.
- Check `%LOCALAPPDATA%\Microsoft\Power BI Desktop\AnalysisServicesWorkspaces\`
  has at least one `AnalysisServicesWorkspace*\Data\msmdsrv.port.txt` file.
- If that file exists but `read_model_summary` fails, the ADOMD.NET MSIs
  probably aren't installed yet.

## How the helper degrades

If any of the above is missing, `start_powerbi.py` still runs — the
"Power BI:" line at startup will say `model-aware mode disabled` and the
UI's top-right pill will read `No .pbix detected`. The LLM still answers
Power BI questions; it just won't reference your real column names.
