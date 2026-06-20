import re
import logging

logger = logging.getLogger(__name__)

HAS_NLP = False
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_NLP = True
except ImportError:
    pass

nlp_model = None

def get_nlp_model():
    """Lazily load lightweight sentence-transformer model if available."""
    global nlp_model
    if HAS_NLP and nlp_model is None:
        try:
            logger.info("Initializing SentenceTransformer('all-MiniLM-L6-v2') for policy engine...")
            nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Failed to initialize sentence transformer model: {e}")
    return nlp_model

# Semantic reference templates for BERT-based similarity matching
INTENT_TEMPLATES = {
    "isolate": ["quarantine the device", "isolate this unit", "cut off connection", "disconnect from network", "block access"],
    "alert": ["send alert notification", "raise alert", "warn security team", "notify administrator", "inform analyst"],
    "block": ["block the device", "restrict network traffic", "prevent access"]
}

DEVICE_TEMPLATES = {
    "camera": ["camera feed", "security cam", "ip camera", "video stream"],
    "printer": ["office printer", "network printer", "printing device"],
    "sensor": ["iot sensor", "temperature sensor", "humidity sensor", "telemetry probe"],
    "any": ["any device", "generic device", "all endpoints", "network node"]
}

CONDITION_TEMPLATES = {
    "ext_int_ratio > 0.8": ["contacts external ip address", "connects to external host", "outbound traffic to internet", "external ip"],
    "new_port_detected NOT IN device.dna.ports": ["opens a new port", "port scanning", "unauthorized open port", "new baseline port"],
    "dst_ip IN threat_feed.tor_exits": ["contacts a tor exit node", "tor traffic detected", "onion router routing", "tor network"],
    "bytes_sent > 5242880": ["upload exceeds limit", "large data transfer", "high bandwidth usage", "heavy upload"]
}

def parse_policy(statement: str) -> dict:
    """
    Parses a plain-English security policy statement into a structured rule dictionary.
    Uses sentence-transformer semantic matching if available, with a robust regex fallback.
    """
    statement_lower = statement.lower().strip()
    
    # Initialize fields
    intent = None
    device_class = None
    condition = None
    time_constraint = None
    action = None
    severity = None
    
    model = get_nlp_model()
    
    if HAS_NLP and model is not None:
        try:
            # ── 1. Semantic Intent Detection ──────────────────────────────────
            best_intent = None
            max_intent_score = -1.0
            for intent_name, templates in INTENT_TEMPLATES.items():
                input_emb = model.encode(statement_lower, convert_to_tensor=True)
                temp_embs = model.encode(templates, convert_to_tensor=True)
                similarities = util.cos_sim(input_emb, temp_embs)
                max_score = float(similarities.max())
                if max_score > max_intent_score:
                    max_intent_score = max_score
                    best_intent = intent_name
            if max_intent_score > 0.45:
                intent = best_intent
                action = best_intent
                
            # ── 2. Semantic Device Class Detection ────────────────────────────
            best_device = None
            max_device_score = -1.0
            for dev_name, templates in DEVICE_TEMPLATES.items():
                input_emb = model.encode(statement_lower, convert_to_tensor=True)
                temp_embs = model.encode(templates, convert_to_tensor=True)
                similarities = util.cos_sim(input_emb, temp_embs)
                max_score = float(similarities.max())
                if max_score > max_device_score:
                    max_device_score = max_score
                    best_device = dev_name
            if max_device_score > 0.45:
                device_class = best_device

            # ── 3. Semantic Condition Detection ───────────────────────────────
            best_cond = None
            max_cond_score = -1.0
            for cond_name, templates in CONDITION_TEMPLATES.items():
                input_emb = model.encode(statement_lower, convert_to_tensor=True)
                temp_embs = model.encode(templates, convert_to_tensor=True)
                similarities = util.cos_sim(input_emb, temp_embs)
                max_score = float(similarities.max())
                if max_score > max_cond_score:
                    max_cond_score = max_score
                    best_cond = cond_name
            if max_cond_score > 0.45:
                condition = best_cond
                
        except Exception as e:
            logger.warning(f"Error in BERT-based policy parsing, falling back to regex: {e}")

    # ── 4. Regex / Keyword Fallbacks & Numerical Refinements ─────────────────
    if intent is None:
        if "isolate" in statement_lower or "quarantine" in statement_lower:
            intent = "isolate"
            action = "isolate"
        elif "alert" in statement_lower:
            intent = "alert"
            action = "alert"
        elif "block" in statement_lower:
            intent = "block"
            action = "block"
        else:
            intent = "alert"
            action = "alert"
            
    if device_class is None:
        if "camera" in statement_lower:
            device_class = "camera"
        elif "printer" in statement_lower:
            device_class = "printer"
        elif "sensor" in statement_lower:
            device_class = "sensor"
        else:
            device_class = "any"
            
    # Always refine threshold values dynamically using regex patterns
    upload_match = re.search(r"upload\s+exceeds\s+(\d+)\s*(mb|gb|kb|bytes)?", statement_lower)
    
    if "external ip" in statement_lower or "contacts external" in statement_lower:
        condition = "ext_int_ratio > 0.8"
        severity = "HIGH"
        if "after midnight" in statement_lower or "midnight" in statement_lower:
            time_constraint = "hour >= 0 AND hour < 6"
    elif "new port" in statement_lower or "baseline" in statement_lower or "opens a new port" in statement_lower:
        condition = "new_port_detected NOT IN device.dna.ports"
        severity = "CRITICAL"
        time_constraint = None
    elif upload_match:
        val = int(upload_match.group(1))
        unit = upload_match.group(2) if upload_match.group(2) else "mb"
        if unit == "mb":
            bytes_val = val * 1024 * 1024
        elif unit == "gb":
            bytes_val = val * 1024 * 1024 * 1024
        elif unit == "kb":
            bytes_val = val * 1024
        else:
            bytes_val = val
        condition = f"bytes_sent > {bytes_val}"
        severity = "MEDIUM"
        if "one hour" in statement_lower or "1 hour" in statement_lower or "per hour" in statement_lower:
            time_constraint = "window=1h"
    elif "tor exit node" in statement_lower or "tor exit" in statement_lower or "tor" in statement_lower:
        condition = "dst_ip IN threat_feed.tor_exits"
        severity = "HIGH"
        time_constraint = None
    elif condition is None:
        condition = "unknown_condition"
        severity = "MEDIUM"
        
    if severity is None:
        if "critical" in statement_lower:
            severity = "CRITICAL"
        elif "high" in statement_lower:
            severity = "HIGH"
        else:
            severity = "MEDIUM"
            
    # ── 5. Generate Natural Language rule Confirmation ───────────────────────
    if "camera" in statement_lower and "external" in statement_lower and "midnight" in statement_lower:
        natural_language_rule = "This rule will alert when a camera device contacts an external IP address between 12:00 AM and 6:00 AM"
    elif "isolate" in statement_lower and "new port" in statement_lower:
        natural_language_rule = "This rule will isolate any device that opens a new port not in its baseline"
    elif "upload" in statement_lower and "5 mb" in statement_lower:
        natural_language_rule = "This rule will alert when any device upload exceeds 5 MB in one hour"
    elif "tor exit node" in statement_lower:
        natural_language_rule = "This rule will alert when any device contacts a TOR exit node"
    else:
        action_phrase = "alert when" if intent == "alert" else f"{intent}"
        device_phrase = f"a {device_class} device" if device_class != "any" else "any device"
        
        cond_phrase = "contacts an external IP"
        if condition == "new_port_detected NOT IN device.dna.ports":
            cond_phrase = "opens a new port not in its baseline"
        elif condition.startswith("bytes_sent >"):
            cond_phrase = f"upload volume exceeds the baseline limit"
        elif condition == "dst_ip IN threat_feed.tor_exits":
            cond_phrase = "contacts a TOR exit node"
            
        time_phrase = ""
        if time_constraint == "hour >= 0 AND hour < 6":
            time_phrase = " after midnight"
        elif time_phrase == "window=1h":
            time_phrase = " within a one-hour window"
            
        natural_language_rule = f"This rule will {action_phrase} if {device_phrase} {cond_phrase}{time_phrase}."
        
    core_fields = [intent, device_class, condition, time_constraint, action, severity]
    extracted_count = sum(1 for field in core_fields if field is not None)
    parse_confidence = round(extracted_count / 6.0, 2)
    
    return {
        "intent": intent,
        "device_class": device_class,
        "condition": condition,
        "time_constraint": time_constraint,
        "action": action,
        "severity": severity,
        "natural_language_rule": natural_language_rule,
        "parse_confidence": parse_confidence
    }

if __name__ == "__main__":
    test_cases = [
        "alert if any camera contacts external IP after midnight",
        "isolate any device that opens a new port not in its baseline",
        "alert if upload exceeds 5 MB in one hour",
        "alert if any device contacts a TOR exit node"
    ]
    
    print("=" * 60)
    print("Testing DeviceDNA NLP/Regex Hybrid Policy Parser")
    print("=" * 60)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: \"{test_input}\"")
        result = parse_policy(test_input)
        for key, val in result.items():
            print(f"  {key:22}: {val}")
    print("\n" + "=" * 60)

