"""Anomaly explanation, pattern detection, and risk forecast (Faza D–E)."""

from __future__ import annotations

from typing import Any, Optional


def explain_anomaly(analysis: dict[str, Any], *, models_listing: Optional[dict] = None) -> dict[str, Any]:
    """Deep root-cause style explanation from tab analysis + optional model list."""
    if not analysis or analysis.get("status") == "tab_not_found":
        return {"ok": False, "error": "tab_not_found", "analysis": analysis}
    snap = analysis.get("snapshot") or {}
    anomaly = analysis.get("anomaly") or {}
    drift = analysis.get("drift") or {}
    events = analysis.get("recent_events") or []

    causes = []
    health = str(anomaly.get("health_state") or snap.get("health_state") or "")
    if snap.get("monitoring_active") is False:
        causes.append("Monitoring is not active — start monitoring before trusting health.")
    if drift.get("is_drift"):
        feats = drift.get("drifted_features") or []
        causes.append(
            f"Concept drift detected (score={drift.get('drift_score')}); features={feats[:8]}"
        )
    if anomaly.get("obs_ok") is False:
        causes.append(f"OBS violations: {anomaly.get('obs_violations')}")
    fusion = anomaly.get("fusion_score")
    if fusion is not None:
        try:
            if float(fusion) >= 0.5:
                causes.append(f"Elevated fusion score ({fusion})")
        except Exception:
            pass
    if events:
        last = events[0] if isinstance(events[0], dict) else {}
        if last.get("xai"):
            causes.append(f"Latest XAI: {last.get('xai')}")
        if last.get("anomalies"):
            causes.append(f"Per-model anomaly counts: {last.get('anomalies')}")

    model_notes = []
    if models_listing and models_listing.get("ok"):
        for m in models_listing.get("models") or []:
            if not m.get("trained"):
                model_notes.append(f"{m.get('model_type')} not trained")
            elif m.get("metrics"):
                model_notes.append(f"{m.get('model_type')} metrics={m.get('metrics')}")

    recommendations = []
    if snap.get("monitoring_active") is False:
        recommendations.append("propose_start_monitoring")
    if drift.get("is_drift"):
        recommendations.append("propose_retrain or propose_train")
    if fusion is not None:
        try:
            if float(fusion) >= 0.5:
                recommendations.append("propose_alert for operator awareness")
        except Exception:
            pass
    if not recommendations:
        recommendations.append("Continue monitoring; no elevated action required")

    return {
        "ok": True,
        "status": "ok",
        "tab_id": analysis.get("tab_id"),
        "title": analysis.get("title") or snap.get("title"),
        "health_state": health,
        "fusion_score": fusion,
        "root_causes": causes or ["No elevated root-cause signals in last cycle"],
        "model_notes": model_notes,
        "recent_event_count": len(events),
        "recommended_actions": recommendations,
        "observation": f"{analysis.get('title') or analysis.get('tab_id')}: health={health}, fusion={fusion}, drift={bool(drift.get('is_drift'))}",
        "analysis": "; ".join(causes) if causes else "Nominal / insufficient cycle data",
        "recommendation": "; ".join(recommendations),
    }


def _autocorr_lag(hist: list[float], lag: int) -> Optional[float]:
    n = len(hist)
    if lag <= 0 or lag >= n // 2:
        return None
    mean = sum(hist) / n
    num = 0.0
    den = 0.0
    for i, x in enumerate(hist):
        d = x - mean
        den += d * d
        if i + lag < n:
            num += d * (hist[i + lag] - mean)
    if den <= 1e-12:
        return None
    return num / den


def _periodicity_hint(hist: list[float]) -> Optional[dict[str, Any]]:
    """Lightweight dominant-lag hint (not a full seasonal model)."""
    if len(hist) < 12:
        return None
    best_lag = None
    best_r = 0.0
    max_lag = min(len(hist) // 3, 48)
    for lag in range(3, max_lag + 1):
        r = _autocorr_lag(hist, lag)
        if r is None:
            continue
        if r > best_r:
            best_r = r
            best_lag = lag
    if best_lag is None or best_r < 0.35:
        return None
    return {
        "type": "periodicity_hint",
        "lag": best_lag,
        "autocorr": round(best_r, 4),
        "method": "heuristic_autocorr",
        "note": "Heuristic lag hint — not a fitted seasonal model (Prophet optional).",
    }


def _seasonal_shift_hint(hist: list[float]) -> Optional[dict[str, Any]]:
    """Flag eclipse-like / seasonal level shifts between early vs late windows."""
    if len(hist) < 10:
        return None
    third = max(3, len(hist) // 3)
    early = hist[:third]
    late = hist[-third:]
    mid = hist[third:-third] if len(hist) > 2 * third else hist[third:]
    mean_e = sum(early) / len(early)
    mean_l = sum(late) / len(late)
    mean_m = sum(mid) / len(mid) if mid else mean_e
    delta = mean_l - mean_e
    # Mid-window dip then recovery → eclipse-like; sustained level change → seasonal shift
    mid_dip = mean_m < min(mean_e, mean_l) - 0.08
    if abs(delta) >= 0.12 or mid_dip:
        kind = "eclipse_like_dip" if mid_dip else "seasonal_level_shift"
        return {
            "type": kind,
            "early_mean": round(mean_e, 4),
            "mid_mean": round(mean_m, 4),
            "late_mean": round(mean_l, 4),
            "delta": round(delta, 4),
            "method": "heuristic_window_means",
            "note": "Heuristic seasonal/eclipse hint for operator review (not Prophet).",
        }
    return None


def detect_patterns(
    score_history: Optional[list] = None,
    events: Optional[list] = None,
    value_history: Optional[list] = None,
) -> dict[str, Any]:
    """Trend / spike / plateau / periodicity heuristics on fusion (and optional value) history."""
    hist = [float(x) for x in (score_history or []) if x is not None]
    patterns: list[dict[str, Any]] = []
    trend = "insufficient_data"
    spike_rate = 0.0

    if len(hist) >= 5:
        first = sum(hist[: max(2, len(hist) // 3)]) / max(1, len(hist) // 3)
        last = sum(hist[-max(2, len(hist) // 3) :]) / max(1, len(hist) // 3)
        delta = last - first
        if delta > 0.05:
            trend = "rising"
            patterns.append({"type": "rising_fusion", "delta": round(delta, 4)})
        elif delta < -0.05:
            trend = "falling"
            patterns.append({"type": "falling_fusion", "delta": round(delta, 4)})
        else:
            trend = "stable"

        sorted_h = sorted(hist)
        med = sorted_h[len(sorted_h) // 2]
        spikes = [x for x in hist if x > med + 0.15]
        spike_rate = round(len(spikes) / len(hist), 4)
        if hist[-1] > med + 0.15:
            patterns.append({"type": "recent_spike", "value": hist[-1], "median": med})
        if spike_rate >= 0.25:
            patterns.append({"type": "elevated_spike_rate", "rate": spike_rate})

        # Plateau: low variance in recent window
        recent = hist[-min(8, len(hist)) :]
        if len(recent) >= 5:
            r_mean = sum(recent) / len(recent)
            var = sum((x - r_mean) ** 2 for x in recent) / len(recent)
            if var < 0.002 and r_mean >= 0.35:
                patterns.append(
                    {
                        "type": "elevated_plateau",
                        "mean": round(r_mean, 4),
                        "variance": round(var, 6),
                    }
                )

        period = _periodicity_hint(hist)
        if period:
            patterns.append(period)
        seasonal = _seasonal_shift_hint(hist)
        if seasonal:
            patterns.append(seasonal)

    # Optional telemetry value series (same heuristics, tagged)
    vals = [float(x) for x in (value_history or []) if x is not None]
    if len(vals) >= 12:
        v_seasonal = _seasonal_shift_hint(vals)
        if v_seasonal:
            tagged = dict(v_seasonal)
            tagged["type"] = f"value_{tagged['type']}"
            tagged["series"] = "value"
            patterns.append(tagged)
        v_period = _periodicity_hint(vals)
        if v_period:
            tagged = dict(v_period)
            tagged["type"] = "value_periodicity_hint"
            tagged["series"] = "value"
            patterns.append(tagged)

    warn_count = 0
    for e in events or []:
        if not isinstance(e, dict):
            continue
        h = str(e.get("health_state") or "").lower()
        if h in ("warning", "critical", "alert"):
            warn_count += 1
    if warn_count >= 3:
        patterns.append({"type": "recurring_warnings", "count": warn_count})

    seasonal_flags = [
        p for p in patterns if "seasonal" in p.get("type", "") or "eclipse" in p.get("type", "")
    ]
    summary = (
        f"Trend={trend}; patterns={len(patterns)}; spike_rate={spike_rate}"
        + (f"; seasonal_hints={len(seasonal_flags)}" if seasonal_flags else "")
        if hist or events or vals
        else "No score history/events yet"
    )

    return {
        "ok": True,
        "status": "ok",
        "trend": trend,
        "history_len": len(hist),
        "spike_rate": spike_rate,
        "patterns": patterns,
        "seasonal_hints": seasonal_flags,
        "summary": summary,
        "method_note": (
            "Heuristic patterns (window means, autocorr lag). "
            "Not a fitted seasonal model unless Prophet is trained on the tab."
        ),
    }


def forecast_risk(
    *,
    forecast_value: Any = None,
    patterns: Optional[dict] = None,
    anomaly: Optional[dict] = None,
    drift: Optional[dict] = None,
) -> dict[str, Any]:
    """Combine short forecast + patterns into a risk score and actions."""
    risk = 0.0
    reasons = []
    anom = anomaly or {}
    dr = drift or {}
    pats = (patterns or {}).get("patterns") or []

    try:
        fusion = float(anom.get("fusion_score")) if anom.get("fusion_score") is not None else None
    except Exception:
        fusion = None
    if fusion is not None and fusion >= 0.7:
        risk += 0.4
        reasons.append(f"high fusion ({fusion})")
    elif fusion is not None and fusion >= 0.45:
        risk += 0.2
        reasons.append(f"moderate fusion ({fusion})")

    if dr.get("is_drift"):
        risk += 0.25
        reasons.append("drift")

    seasonal_ops_note = ""
    for p in pats:
        ptype = str(p.get("type") or "")
        if ptype == "rising_fusion":
            risk += 0.15
            reasons.append("rising fusion trend")
        if ptype == "recent_spike":
            risk += 0.15
            reasons.append("recent spike")
        if ptype == "recurring_warnings":
            risk += 0.2
            reasons.append("recurring warnings")
        if ptype == "elevated_spike_rate":
            risk += 0.1
            reasons.append(f"elevated spike rate ({p.get('rate')})")
        if ptype == "elevated_plateau":
            risk += 0.1
            reasons.append("elevated fusion plateau")
        if "seasonal" in ptype or "eclipse" in ptype:
            risk += 0.1
            reasons.append(f"seasonal/eclipse hint ({ptype})")
            seasonal_ops_note = (
                "Seasonal/eclipse-like level shift hinted (heuristic). "
                "Cross-check SOP eclipse/battery procedures; Prophet seasonal fit only if that model is trained."
            )
        if "periodicity" in ptype:
            reasons.append(f"periodicity hint lag={p.get('lag')}")

    try:
        if forecast_value is not None and float(forecast_value) >= 0.5:
            risk += 0.15
            reasons.append(f"short-horizon forecast elevated ({forecast_value})")
    except Exception:
        pass

    risk = min(1.0, round(risk, 3))
    if risk >= 0.7:
        level = "HIGH"
        actions = ["propose_alert CRITICAL", "propose_retrain", "review SOP response"]
    elif risk >= 0.4:
        level = "MEDIUM"
        actions = ["propose_alert WARNING", "request_monitor_cycle", "compare_tab_models"]
    else:
        level = "LOW"
        actions = ["continue monitoring"]

    if seasonal_ops_note and "review SOP" not in "; ".join(actions):
        actions.append("review SOP for seasonal/eclipse ops")

    observation = (
        f"risk={risk} ({level}); forecast={forecast_value}; "
        f"pattern_count={len(pats)}"
    )
    analysis = "; ".join(reasons) if reasons else "no elevated signals"
    if seasonal_ops_note:
        analysis = analysis + ". " + seasonal_ops_note

    return {
        "ok": True,
        "status": "ok",
        "risk_score": risk,
        "risk_level": level,
        "reasons": reasons or ["no elevated signals"],
        "suggested_actions": actions,
        "forecast_value": forecast_value,
        "what_to_do": "; ".join(actions),
        "seasonal_note": seasonal_ops_note or None,
        "observation": observation,
        "analysis": analysis,
        "recommendation": "; ".join(actions),
        "method_note": (
            "Scorecard over fusion/drift/patterns/short forecast — not a full seasonal forecaster."
        ),
    }
