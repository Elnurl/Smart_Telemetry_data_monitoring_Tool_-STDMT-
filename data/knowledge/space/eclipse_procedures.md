# Eclipse Operations (space)

## Entry
1. Confirm eclipse entry in mission/event logs (TCS / POWER).
2. Switch the monitoring tab Mission Mode to **eclipse** (threshold_scale ≈ 1.5) if not auto-detected.
3. Expect battery temperature and bus current shape changes; do not treat typical TCS WARNINGs as anomalies.

## During eclipse
- Watch for CDH / command errors — those remain true anomalies in every mode.
- Fusion threshold is widened by threshold_scale; mode-normal OBS hits are recorded separately.

## Exit
1. On eclipse exit / sunlight mode, return Mission Mode to **nominal**.
2. Re-check thermal recovery; residual out-of-family values after exit may be real anomalies.
