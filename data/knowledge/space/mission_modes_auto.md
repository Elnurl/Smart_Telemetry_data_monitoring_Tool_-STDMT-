# Mission Modes (auto-exported from STDMS tab FSM)

Generated for RAG space store. Do not edit by hand — Rebuild Knowledge refreshes this file.

# Tab: Battery Temperature Monitoring — Mission Modes
- tab_id: `0356477f-54f2-4172-b1a0-1713640e6ee4`
- current_mission_mode: **nominal**

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

# Tab: test — Mission Modes
- tab_id: `57738f17-2cb3-421b-861a-72c16b7394b6`
- current_mission_mode: **nominal**

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
