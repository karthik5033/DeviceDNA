"""
Feature-to-Language Mapping Module
===================================
Converts raw SHAP explainability results into human-readable security
statements suitable for SOC analyst consumption.

Each of the 14 telemetry features has two templates — one for an anomalous
*increase* and one for an anomalous *decrease* — giving 28 total entries.

Usage:
    from app.ml.explainability.feature_language import feature_to_statement

    statement = feature_to_statement({
        "feature_name": "burst_freq",
        "shap_value": -0.75,
        "observed_value": 0.1,
        "baseline_value": 0.02,
        "direction": "increase"
    })
"""

# ──────────────────────────────────────────────────────────────────────
# Lookup table:  (feature_name, direction) → template string
#
# Templates use {observed} and {baseline} as interpolation slots.
# They are written in the voice of a concise SOC alert explanation.
# ──────────────────────────────────────────────────────────────────────

_FEATURE_TEMPLATES: dict[tuple[str, str], str] = {

    # ── bytes_sent ────────────────────────────────────────────────────
    ("bytes_sent", "increase"):
        "Outbound data volume surged to {observed} bytes (baseline: {baseline}) "
        "— device is transmitting significantly more data than its normal profile",
    ("bytes_sent", "decrease"):
        "Outbound data volume dropped to {observed} bytes (baseline: {baseline}) "
        "— device may be throttled, isolated, or entering a dormant C2 staging phase",

    # ── bytes_recv ────────────────────────────────────────────────────
    ("bytes_recv", "increase"):
        "Inbound data volume surged to {observed} bytes (baseline: {baseline}) "
        "— device is receiving abnormally large payloads, possible firmware push or exfiltration relay",
    ("bytes_recv", "decrease"):
        "Inbound data volume dropped to {observed} bytes (baseline: {baseline}) "
        "— device may have lost upstream connectivity or is being starved by a network disruption",

    # ── packet_count ──────────────────────────────────────────────────
    ("packet_count", "increase"):
        "Packet count spiked to {observed} (baseline: {baseline}) "
        "— abnormal burst of network activity suggesting scanning or flooding behavior",
    ("packet_count", "decrease"):
        "Packet count fell to {observed} (baseline: {baseline}) "
        "— unusually quiet device activity, potential indicator of pre-attack dormancy or link failure",

    # ── bytes_per_packet ──────────────────────────────────────────────
    ("bytes_per_packet", "increase"):
        "Average packet size grew to {observed} bytes (baseline: {baseline}) "
        "— oversized packets may indicate data exfiltration or tunneling attempts",
    ("bytes_per_packet", "decrease"):
        "Average packet size shrank to {observed} bytes (baseline: {baseline}) "
        "— small packets are consistent with beaconing, keep-alive, or C2 heartbeat patterns",

    # ── upload_download_ratio ─────────────────────────────────────────
    ("upload_download_ratio", "increase"):
        "Upload/download ratio rose to {observed} (baseline: {baseline}) "
        "— device is sending disproportionately more than it receives, consistent with data exfiltration",
    ("upload_download_ratio", "decrease"):
        "Upload/download ratio dropped to {observed} (baseline: {baseline}) "
        "— possible data staging or receive-heavy C2 pattern",

    # ── unique_dst_ips ────────────────────────────────────────────────
    ("unique_dst_ips", "increase"):
        "Device contacted {observed} unique destination IPs (baseline: {baseline}) "
        "— abnormal external reach suggesting reconnaissance or lateral movement",
    ("unique_dst_ips", "decrease"):
        "Unique destination IPs fell to {observed} (baseline: {baseline}) "
        "— device has narrowed communication to fewer endpoints, possible C2 lock-in",

    # ── unique_dst_ports ──────────────────────────────────────────────
    ("unique_dst_ports", "increase"):
        "Device reached {observed} unique destination ports (baseline: {baseline}) "
        "— port fanning detected, consistent with service scanning or exploitation attempts",
    ("unique_dst_ports", "decrease"):
        "Unique destination ports fell to {observed} (baseline: {baseline}) "
        "— device has locked onto fewer services, possible single-target attack focus",

    # ── unique_src_ports ──────────────────────────────────────────────
    ("unique_src_ports", "increase"):
        "Source port count rose to {observed} (baseline: {baseline}) "
        "— high ephemeral port churn may indicate rapid connection cycling or tunneling",
    ("unique_src_ports", "decrease"):
        "Source port count dropped to {observed} (baseline: {baseline}) "
        "— low port diversity is unusual and may indicate a fixed-port backdoor channel",

    # ── ext_int_ratio ─────────────────────────────────────────────────
    ("ext_int_ratio", "increase"):
        "External traffic ratio rose to {observed} (baseline: {baseline}) "
        "— unusually high outbound communication, potential data leak to external hosts",
    ("ext_int_ratio", "decrease"):
        "External traffic ratio dropped to {observed} (baseline: {baseline}) "
        "— device is communicating almost entirely within the LAN, possible lateral movement pivot",

    # ── active_hours_bitmap ───────────────────────────────────────────
    ("active_hours_bitmap", "increase"):
        "Device active outside normal hours — activity window expanded to bitmap {observed} "
        "(baseline: {baseline}), indicating operation beyond its expected schedule",
    ("active_hours_bitmap", "decrease"):
        "Device activity window contracted to bitmap {observed} (baseline: {baseline}) "
        "— narrower-than-normal operating hours may indicate selective covert timing",

    # ── inter_arrival_mean ────────────────────────────────────────────
    ("inter_arrival_mean", "increase"):
        "Mean inter-arrival time increased to {observed}ms (baseline: {baseline}ms) "
        "— packets are arriving slower than expected, possible throttled exfiltration or link degradation",
    ("inter_arrival_mean", "decrease"):
        "Mean inter-arrival time dropped to {observed}ms (baseline: {baseline}ms) "
        "— rapid-fire packet cadence consistent with automated botnet communication or DDoS participation",

    # ── inter_arrival_var ─────────────────────────────────────────────
    ("inter_arrival_var", "increase"):
        "Inter-arrival time variance spiked to {observed} (baseline: {baseline}) "
        "— erratic timing pattern suggests jittered C2 beaconing designed to evade periodic detection",
    ("inter_arrival_var", "decrease"):
        "Inter-arrival time variance dropped to {observed} (baseline: {baseline}) "
        "— highly uniform packet timing is a strong indicator of automated or scripted traffic",

    # ── burst_freq ────────────────────────────────────────────────────
    ("burst_freq", "increase"):
        "Traffic burst frequency spiked to {observed} (baseline: {baseline}) "
        "— device is sending data in abnormal rapid bursts",
    ("burst_freq", "decrease"):
        "Traffic burst frequency fell to {observed} (baseline: {baseline}) "
        "— unusually steady flow replaces normal bursty behavior, consistent with persistent tunnel or stream",

    # ── protocol_entropy ──────────────────────────────────────────────
    ("protocol_entropy", "increase"):
        "Protocol entropy rose to {observed} (baseline: {baseline}) "
        "— device is using an unusually diverse mix of protocols, possible evasion or multi-vector attack",
    ("protocol_entropy", "decrease"):
        "Protocol entropy dropped to {observed} (baseline: {baseline}) "
        "— device has collapsed to a single protocol, possible indicator of a dedicated covert channel",
}

_FALLBACK_TEMPLATE = (
    "Anomalous change detected in {feature_name} "
    "(observed: {observed}, baseline: {baseline})"
)


def feature_to_statement(shap_result: dict) -> str:
    """
    Convert a single SHAP result dict into a human-readable security statement.

    Parameters
    ----------
    shap_result : dict
        Must contain at minimum:
            - feature_name  (str)
            - observed_value (float)
            - baseline_value (float)
            - direction      (str, "increase" or "decrease")

    Returns
    -------
    str
        A filled security-analyst-grade explanation sentence.
    """
    feature_name = shap_result.get("feature_name", "unknown")
    direction = shap_result.get("direction", "increase")
    observed = shap_result.get("observed_value", "N/A")
    baseline = shap_result.get("baseline_value", "N/A")

    # Format numeric values to 4 decimal places if they are floats
    if isinstance(observed, float):
        observed = f"{observed:.4f}"
    if isinstance(baseline, float):
        baseline = f"{baseline:.4f}"

    key = (feature_name, direction)
    template = _FEATURE_TEMPLATES.get(key, None)

    if template is None:
        return _FALLBACK_TEMPLATE.format(
            feature_name=feature_name,
            observed=observed,
            baseline=baseline,
        )

    return template.format(observed=observed, baseline=baseline)


# ──────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulated SHAP results from shap_engine
    test_cases = [
        {"feature_name": "burst_freq", "shap_value": -0.75, "observed_value": 0.1, "baseline_value": 0.02, "direction": "increase"},
        {"feature_name": "unique_dst_ips", "shap_value": 0.60, "observed_value": 12.0, "baseline_value": 2.0, "direction": "increase"},
        {"feature_name": "upload_download_ratio", "shap_value": -0.65, "observed_value": 0.05, "baseline_value": 0.4, "direction": "decrease"},
        {"feature_name": "active_hours_bitmap", "shap_value": 0.38, "observed_value": 255.0, "baseline_value": 60.0, "direction": "increase"},
        {"feature_name": "protocol_entropy", "shap_value": -0.70, "observed_value": 0.1, "baseline_value": 1.8, "direction": "decrease"},
        # Fallback test — unknown feature
        {"feature_name": "mystery_metric", "shap_value": 0.42, "observed_value": 99.0, "baseline_value": 1.0, "direction": "increase"},
    ]

    print("=" * 80)
    print("FEATURE-TO-LANGUAGE MAPPING — SELF-TEST")
    print("=" * 80)

    for i, case in enumerate(test_cases, 1):
        stmt = feature_to_statement(case)
        tag = f"[{case['feature_name']} / {case['direction']}]"
        print(f"\n{i}. {tag}")
        print(f"   -> {stmt}")

    print(f"\n{'=' * 80}")
    print(f"Lookup table coverage: {len(_FEATURE_TEMPLATES)} entries (14 features × 2 directions)")
    print(f"{'=' * 80}")
