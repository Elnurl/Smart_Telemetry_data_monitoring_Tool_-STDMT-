# Battery Temperature Monitoring (space)

## Purpose
Monitor satellite battery temperature telemetry for safe electrical power subsystem (EPS) operation.

## Process
1. Collect temperature samples from the battery / thermal sensors.
2. Analyze for anomalies against the active mission mode thresholds.
3. In **eclipse**, elevated thermal deviation may be mode-normal (see mission_modes.md).
4. In **nominal**, temperature exceeding base limits is an anomaly candidate.
5. Document incidents and escalate CDH / critical electrical faults immediately.

## Related
- Mission modes FSM (threshold_scale)
- Eclipse procedures
