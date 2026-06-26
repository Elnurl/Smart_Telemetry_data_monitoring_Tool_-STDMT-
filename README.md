# STDMS v3.1 — Satellite Telemetry Data Monitoring Tool

PyQt desktop app: custom monitoring tabs, anomaly detection, fleet dashboard.

## License

MIT License — see [LICENSE](LICENSE).

## Run (Python 3.10 only)

```powershell
cd STDMS_3.1
.\run.bat
```

Or: `py -3.10 main.py`

Do **not** use `python main.py` if default is Python 3.14.

## Setup

```powershell
py -3.10 -m pip install -r requirements.txt
copy data\email_config.example.json data\email_config.json
```

## Tests

```powershell
py -3.10 -m pytest tests/unit -q
```
