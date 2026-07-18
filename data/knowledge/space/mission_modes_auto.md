# Mission Modes (auto-exported from STDMS tab FSM)

Generated for RAG space store. Do not edit by hand — Rebuild Knowledge refreshes this file.

# Tab: Battery Temperature Monitoring — Mission Modes
- tab_id: `98dfc738-9a64-4148-997a-18d51f6dee70`
- current_mission_mode: **eclipse**

## nominal (threshold_scale: 1)
- Normal operations
- All deviations against base thresholds are anomalies

## eclipse (threshold_scale: 1.5)
- Eclipse — thermal swings expected
- TCS WARNING / thermal swing → mode-normal (expected in eclipse)
- Solar panel / battery telemetry deviation may be expected

## maneuver (threshold_scale: 2)
- Maneuver — vibration / transient spikes expected
- ADCS / vibration transients → often mode-normal

## safe_mode (threshold_scale: 0.5)
- Safe mode — tighter sensitivity
- Tighter sensitivity (scale < 1); escalate warnings faster

---
