# Mission Modes (space segment)

STDMS tabs can run under different satellite operating regimes. Threshold scaling changes how OBS and fusion treat telemetry swings.

## nominal (threshold_scale: 1.0)
- Baseline operations
- All deviations against base thresholds are anomalies

## eclipse (threshold_scale: 1.5)
- Thermal / battery temperature swings are often expected
- TCS WARNING about temperature exceeding nominal range → mode-normal when eclipse is active
- Eclipse entry is usually logged by TCS / POWER

## maneuver (threshold_scale: 2.0)
- Attitude control transients and vibration may look like outliers
- ADCS-related WARNINGs are often mode-normal during the burn / slew

## safe_mode (threshold_scale: 0.5)
- Tighter sensitivity — escalate sooner
- Prefer operator confirmation before clearing elevated health
