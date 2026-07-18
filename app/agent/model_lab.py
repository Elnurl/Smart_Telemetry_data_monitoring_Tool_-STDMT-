"""Model Lab — compare metrics, propose model plans, lightweight optimize hints (Faza B)."""

from __future__ import annotations

import uuid
from typing import Any, Optional


def _score_metrics(metrics: dict[str, Any]) -> float:
    """Higher is better. Prefer precision/recall/f1/auc when present."""
    if not isinstance(metrics, dict) or not metrics:
        return 0.0
    score = 0.0
    weights = {
        "f1": 3.0,
        "f1_score": 3.0,
        "precision": 2.0,
        "recall": 2.0,
        "roc_auc": 2.5,
        "auc": 2.5,
        "accuracy": 1.5,
        "average_precision": 2.0,
    }
    for key, w in weights.items():
        if key in metrics:
            try:
                score += w * float(metrics[key])
            except Exception:
                pass
    # Lower contamination / error-like keys slightly if present as quality
    for key in ("contamination", "mse", "mae", "error"):
        if key in metrics:
            try:
                score -= 0.1 * float(metrics[key])
            except Exception:
                pass
    return round(score, 4)


def compare_tab_models(listing: dict[str, Any]) -> dict[str, Any]:
    """Rank models from list_tab_models output."""
    if not listing or not listing.get("ok"):
        return {
            "ok": False,
            "error": listing.get("error") if isinstance(listing, dict) else "listing_unavailable",
            "models": [],
        }
    rows = []
    for m in listing.get("models") or []:
        metrics = m.get("metrics") or {}
        rows.append(
            {
                "model_id": m.get("model_id"),
                "model_type": m.get("model_type"),
                "trained": bool(m.get("trained")),
                "selected": bool(m.get("selected")),
                "metrics": metrics,
                "score": _score_metrics(metrics) if m.get("trained") else 0.0,
            }
        )
    rows.sort(key=lambda r: (r["trained"], r["score"]), reverse=True)
    best = next((r for r in rows if r.get("trained")), None)
    return {
        "ok": True,
        "status": "ok",
        "tab_id": listing.get("tab_id"),
        "title": listing.get("title"),
        "models": rows,
        "best_model_id": best.get("model_id") if best else None,
        "best_model_type": best.get("model_type") if best else None,
        "best_score": best.get("score") if best else None,
        "recommendation": (
            f"Prefer model {best.get('model_id')} ({best.get('model_type')}) — Approve train/retrain if drift rises."
            if best
            else "No trained models yet — propose_train or propose_model_plan first."
        ),
    }


def propose_model_plan(
    *,
    purpose: str = "health",
    include_types: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Suggest a multi-model plan with default hyperparameters (not applied until Approve train)."""
    purpose_l = (purpose or "").lower()
    defaults = include_types or [
        "Isolation Forest",
        "Local Outlier Factor",
        "Z-Score",
        "Random Forest" if "supervised" in purpose_l or "rf" in purpose_l else None,
    ]
    defaults = [d for d in defaults if d]
    plans = []
    grids = {
        "Isolation Forest": [
            {"n_estimators": 100, "contamination": 0.05},
            {"n_estimators": 200, "contamination": 0.1},
        ],
        "Local Outlier Factor": [
            {"n_neighbors": 15, "contamination": 0.05},
            {"n_neighbors": 30, "contamination": 0.1},
        ],
        "Z-Score": [{"threshold": 3.0}, {"threshold": 2.5}],
        "Random Forest": [
            {"n_estimators": 100, "contamination": 0.1},
            {"n_estimators": 200, "contamination": 0.05},
        ],
    }
    for mt in defaults:
        for params in grids.get(mt, [{}]):
            plans.append(
                {
                    "model_type": mt,
                    "model_parameters": dict(params),
                    "suggested_model_id": str(uuid.uuid4()),
                    "rationale": f"Candidate for {purpose or 'monitoring'}",
                }
            )
    return {
        "ok": True,
        "status": "ok",
        "purpose": purpose,
        "candidates": plans,
        "recommended_workflow": [
            "Add candidates to tab config (propose_config_change or create_tab)",
            "propose_train each suggested_model_id",
            "compare_tab_models → keep best",
        ],
    }


def suggest_optimize_from_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    """Lightweight optimize hint from compare_tab_models (no silent train)."""
    if not comparison or not comparison.get("ok"):
        return {"ok": False, "error": "comparison_unavailable"}
    best_id = comparison.get("best_model_id")
    models = comparison.get("models") or []
    untrained = [m for m in models if not m.get("trained")]
    hints = []
    if best_id:
        hints.append(
            {
                "action": "propose_train",
                "model_id": best_id,
                "message": f"Retrain/keep best model {best_id} after data refresh",
            }
        )
    for m in untrained[:3]:
        hints.append(
            {
                "action": "propose_train",
                "model_id": m.get("model_id"),
                "message": f"Train untrained {m.get('model_type')} ({m.get('model_id')})",
            }
        )
    # Suggest contamination tweak for IF if metrics weak
    for m in models:
        if m.get("trained") and m.get("score", 0) < 1.0 and "Isolation" in str(m.get("model_type") or ""):
            hints.append(
                {
                    "action": "propose_config_change",
                    "model_id": m.get("model_id"),
                    "config_patch": {
                        "models_patch": {
                            "model_id": m.get("model_id"),
                            "model_parameters": {"contamination": 0.08, "n_estimators": 250},
                        }
                    },
                    "message": "Try slightly higher contamination / more trees for Isolation Forest",
                }
            )
            break
    return {
        "ok": True,
        "status": "ok",
        "best_model_id": best_id,
        "optimize_hints": hints,
        "note": "All trains/config changes require human Approve via propose_* drafts.",
    }
