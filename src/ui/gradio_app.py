"""
Gradio UI mounted on the FastAPI app at /ui.

What this module builds:
  - A 9-feature input form (sliders, dropdowns, radio) with 3 one-click presets
    (Normal traffic / Obvious attack / Edge case).
  - A champion verdict banner (big coloured block + probability + risk level).
  - A live SHAP waterfall plot for each prediction, using a TreeExplainer cached
    after the first call.
  - A side-by-side "All 4 candidate models on this input" section: one card per
    model (LR / RF champion / XGBoost / DNN v2) with the per-sample probability,
    the measured inference time, and the model's global F1 on the test set.
    A consensus banner below the cards flags the spread (CONSENSUS / PARTIAL /
    DISAGREEMENT) so the champion choice is defensible even on a single sample
    where the per-sample probabilities happen to be tied.

Integration with FastAPI:
  build_demo() returns a gr.Blocks that the parent app mounts with
  gr.mount_gradio_app(app, build_demo(), path='/ui'). The callbacks lazily import
  the API module to reach `model`, `preprocessor` and `comparison_models`, so
  FastAPI's startup event has finished loading artefacts before the UI ever
  serves a prediction.
"""

from __future__ import annotations

import base64
import io
import time

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils.feature_engineering import add_all_features

matplotlib.use("Agg")


PROTOCOLS = ["TCP", "UDP", "ICMP"]
ENCRYPTIONS = ["AES", "DES", "None"]
BROWSERS = ["Chrome", "Firefox", "Edge", "Safari", "Unknown"]


PRESETS = {
    "Normal traffic": {
        "network_packet_size": 500, "protocol_type": "TCP", "login_attempts": 1,
        "session_duration": 120.0, "encryption_used": "AES", "ip_reputation_score": 0.20,
        "failed_logins": 0, "browser_type": "Chrome", "unusual_time_access": 0,
    },
    "Obvious attack": {
        "network_packet_size": 200, "protocol_type": "UDP", "login_attempts": 10,
        "session_duration": 30.0, "encryption_used": "None", "ip_reputation_score": 0.85,
        "failed_logins": 5, "browser_type": "Unknown", "unusual_time_access": 1,
    },
    "Edge case (likely FN)": {
        "network_packet_size": 800, "protocol_type": "TCP", "login_attempts": 3,
        "session_duration": 400.0, "encryption_used": "AES", "ip_reputation_score": 0.55,
        "failed_logins": 2, "browser_type": "Edge", "unusual_time_access": 0,
    },
}


_explainer = None
_feature_names: list[str] | None = None


def _get_model_and_preprocessor():
    """Import lazily so FastAPI startup can load the bundle first."""
    from src.api import app as api_module

    if api_module.model is None or api_module.preprocessor is None:
        raise RuntimeError("Model not loaded yet — wait for API startup to complete.")
    return api_module.model, api_module.preprocessor


def _get_comparison_models():
    """Return the dict of comparison models (may be empty if export script wasn't run)."""
    from src.api import app as api_module
    return api_module.comparison_models, api_module.comparison_metadata


def _get_explainer(model, preprocessor):
    """TreeExplainer is fast for RF/XGBoost (~ms). Cache after first call."""
    global _explainer, _feature_names
    if _explainer is None:
        _explainer = shap.TreeExplainer(model)
        try:
            _feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            _feature_names = None
    return _explainer, _feature_names


def _waterfall(shap_values_row, feature_names, raw_values) -> plt.Figure:
    """
    Build a waterfall-style horizontal bar chart of feature contributions.
    We roll our own to avoid shap's global plt state and keep the figure
    self-contained for Gradio's gr.Plot output.
    """
    top_k = 10
    order = np.argsort(np.abs(shap_values_row))[::-1][:top_k]
    contribs = shap_values_row[order]
    names = [feature_names[i] if feature_names else f"f{i}" for i in order]
    values = [raw_values[i] for i in order]
    labels = [f"{n} = {v:.2f}" for n, v in zip(names, values)]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
    colors = ["#d62728" if c > 0 else "#1f77b4" for c in contribs]
    y_pos = np.arange(len(contribs))[::-1]
    ax.barh(y_pos, contribs, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("SHAP value (→ push toward Attack, ← push toward Normal)")
    ax.set_title("Top feature contributions for this prediction", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def _proba(model, X) -> float:
    """Unified probability extraction across sklearn / xgboost / keras."""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 1])
    # Keras DNN: sigmoid output
    return float(np.asarray(model.predict(X, verbose=0)).ravel()[0])


def _comparison_html(
    results: list[dict],
    metadata: dict,
    champion_name: str = "Random Forest",
) -> str:
    """
    Build the side-by-side cards HTML for the comparison models section.
    `metadata[name] = {'f1': ..., 'auc': ..., ...}` comes from comparison_metadata.joblib
    and is displayed under each card so the viewer can see the per-prediction probability
    *and* the global test-set F1 together — that's what justifies the champion choice
    even when it's not the highest on the current sample.
    """
    probas = [r["proba"] for r in results]
    spread = max(probas) - min(probas) if probas else 0.0
    if spread < 0.15:
        consensus = ("#2ca02c", "CONSENSUS — all models agree on this session")
    elif spread < 0.35:
        consensus = ("#ff7f0e", "PARTIAL — some disagreement between models")
    else:
        consensus = ("#d62728", "DISAGREEMENT — models split, ambiguous case")

    # Highlight the best global F1 across models (= champion justification)
    best_f1 = max((metadata.get(r["name"], {}).get("f1", 0) for r in results), default=0)

    cards = []
    for r in results:
        color = "#d62728" if r["pred"] == 1 else "#2ca02c"
        label = "Attack" if r["pred"] == 1 else "Normal"
        highlight = "box-shadow: 0 0 0 3px #ffd700;" if r["name"] == champion_name else ""
        m = metadata.get(r["name"], {})
        f1_val = m.get("f1")
        auc_val = m.get("auc")
        f1_color = "#2ca02c" if f1_val is not None and abs(f1_val - best_f1) < 1e-6 else "#555"
        f1_weight = "700" if f1_val is not None and abs(f1_val - best_f1) < 1e-6 else "400"
        f1_line = (
            f"<div style='font-size:11px; margin-top:6px; padding-top:6px; border-top:1px dashed #ccc;'>"
            f"<span style='color:#666;'>F1 (test):</span> "
            f"<b style='color:{f1_color}; font-weight:{f1_weight};'>{f1_val:.3f}</b>"
            f"&nbsp;·&nbsp;<span style='color:#666;'>AUC:</span> {auc_val:.3f}"
            f"</div>"
        ) if f1_val is not None else ""
        cards.append(f"""
        <div style='flex:1; min-width:150px; padding:12px; margin:4px;
                    border:1px solid #ddd; border-radius:8px; {highlight}
                    font-family:monospace; background:#fafafa;'>
          <div style='font-size:12px; color:#666;'>{r['name']}
            {"<b style='color:#b8860b;'> ★ CHAMPION</b>" if r['name'] == champion_name else ""}</div>
          <div style='font-size:22px; font-weight:700; color:{color}; margin:6px 0;'>
            {r['proba']:.3f}
          </div>
          <div style='font-size:13px;'>{label}</div>
          <div style='font-size:11px; color:#888; margin-top:4px;'>
            {r['latency_ms']:.2f} ms
          </div>
          {f1_line}
        </div>
        """)

    return f"""
    <div>
      <div style='display:flex; flex-wrap:wrap; margin-bottom:10px;'>{''.join(cards)}</div>
      <div style='padding:10px; border-radius:6px; background:{consensus[0]}22;
                  border-left:4px solid {consensus[0]}; font-family:monospace; font-size:13px;'>
        <b style='color:{consensus[0]};'>{consensus[1]}</b>
        &nbsp;·&nbsp; spread on this sample: <b>{spread:.3f}</b>
        &nbsp;·&nbsp; champion (RF) chosen on <b>global F1</b>, not per-sample probability.
      </div>
    </div>
    """


def predict_and_explain(
    network_packet_size: int,
    protocol_type: str,
    login_attempts: int,
    session_duration: float,
    encryption_used: str,
    ip_reputation_score: float,
    failed_logins: int,
    browser_type: str,
    unusual_time_access: int,
):
    model, preprocessor = _get_model_and_preprocessor()

    raw = {
        "network_packet_size": int(network_packet_size),
        "protocol_type": protocol_type,
        "login_attempts": int(login_attempts),
        "session_duration": float(session_duration),
        "encryption_used": encryption_used,
        "ip_reputation_score": float(ip_reputation_score),
        "failed_logins": int(failed_logins),
        "browser_type": browser_type,
        "unusual_time_access": int(unusual_time_access),
    }
    df = add_all_features(pd.DataFrame([raw]))
    X = preprocessor.transform(df)

    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    label = "ATTACK DETECTED" if pred == 1 else "NORMAL TRAFFIC"
    color = "#d62728" if pred == 1 else "#2ca02c"

    if proba > 0.8:
        risk = "CRITICAL"
    elif proba > 0.5:
        risk = "HIGH"
    elif proba > 0.3:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    verdict_html = f"""
    <div style='text-align:center; padding:20px; border-radius:10px;
                background-color:{color}; color:white; font-family:monospace;'>
      <div style='font-size:28px; font-weight:700;'>{label}</div>
      <div style='font-size:16px; margin-top:6px;'>
        Champion (Random Forest) — probability: <b>{proba:.4f}</b> &nbsp;·&nbsp; Risk: <b>{risk}</b>
      </div>
    </div>
    """

    explainer, feature_names = _get_explainer(model, preprocessor)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv_row = np.asarray(sv[1])[0] if len(sv) == 2 else np.asarray(sv[0])[0]
    else:
        sv_arr = np.asarray(sv)
        sv_row = sv_arr[0, :, 1] if sv_arr.ndim == 3 else sv_arr[0]

    raw_row = np.asarray(X.toarray() if hasattr(X, "toarray") else X)[0]
    fig = _waterfall(sv_row, feature_names, raw_row)

    # --- Comparison across all 4 models ---
    comp_models, comp_meta = _get_comparison_models()
    results = []
    # Ordered LR → RF → XGB → DNN for consistent reading
    for name in ["Logistic Regression", "Random Forest", "XGBoost", "DNN v2"]:
        m = comp_models.get(name)
        if m is None:
            continue
        t0 = time.perf_counter()
        p = _proba(m, X)
        latency_ms = (time.perf_counter() - t0) * 1000
        results.append({"name": name, "proba": p, "pred": int(p >= 0.5), "latency_ms": latency_ms})

    comparison_html = _comparison_html(results, comp_meta) if results else ""

    return verdict_html, fig, comparison_html


def _apply_preset(name: str):
    p = PRESETS[name]
    return (
        p["network_packet_size"], p["protocol_type"], p["login_attempts"],
        p["session_duration"], p["encryption_used"], p["ip_reputation_score"],
        p["failed_logins"], p["browser_type"], p["unusual_time_access"],
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="IDS — Cybersecurity Intrusion Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Cybersecurity Intrusion Detection\n"
            "**Champion model: Random Forest** (F1=0.855 on held-out test set).\n"
            "Enter a network session below — the champion predicts whether it's an attack, "
            "SHAP explains which features drove the decision, and the 4 candidate models "
            "show their individual probability side-by-side."
        )

        with gr.Row():
            preset_buttons = [gr.Button(name, variant="secondary") for name in PRESETS]

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Session features")
                network_packet_size = gr.Slider(64, 1500, value=500, step=1, label="network_packet_size (bytes)")
                protocol_type = gr.Dropdown(PROTOCOLS, value="TCP", label="protocol_type")
                login_attempts = gr.Slider(0, 20, value=1, step=1, label="login_attempts")
                session_duration = gr.Slider(0, 1000, value=120, step=1, label="session_duration (s)")
                encryption_used = gr.Dropdown(ENCRYPTIONS, value="AES", label="encryption_used")
                ip_reputation_score = gr.Slider(0, 1, value=0.2, step=0.01, label="ip_reputation_score")
                failed_logins = gr.Slider(0, 20, value=0, step=1, label="failed_logins")
                browser_type = gr.Dropdown(BROWSERS, value="Chrome", label="browser_type")
                unusual_time_access = gr.Radio([0, 1], value=0, label="unusual_time_access")

                predict_btn = gr.Button("Predict", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Champion verdict")
                verdict = gr.HTML()
                gr.Markdown("### Why? (SHAP)")
                shap_plot = gr.Plot()

        gr.Markdown("### All 4 candidate models on this input")
        comparison = gr.HTML()

        inputs = [
            network_packet_size, protocol_type, login_attempts, session_duration,
            encryption_used, ip_reputation_score, failed_logins, browser_type,
            unusual_time_access,
        ]

        predict_btn.click(
            fn=predict_and_explain,
            inputs=inputs,
            outputs=[verdict, shap_plot, comparison],
        )

        for btn, name in zip(preset_buttons, PRESETS):
            btn.click(fn=lambda n=name: _apply_preset(n), inputs=None, outputs=inputs)

    return demo
