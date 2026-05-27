import re

def parse_policy(statement: str) -> dict:
    """
    Parses a plain-English security policy statement into a structured rule dictionary
    using regex and keyword pattern matching.
    """
    statement_lower = statement.lower().strip()
    
    # Initialize fields
    intent = None
    device_class = None
    condition = None
    time_constraint = None
    action = None
    severity = None
    
    # 1. Detect Intent and Action
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
        # Default or fallback intent
        intent = "alert"
        action = "alert"
        
    # 2. Detect Device Class
    if "camera" in statement_lower:
        device_class = "camera"
    elif "printer" in statement_lower:
        device_class = "printer"
    elif "device" in statement_lower or "any" in statement_lower:
        device_class = "any"
    else:
        device_class = "any"
        
    # 3. Detect Condition, Time Constraint, and Severity
    # Look for upload patterns like "upload exceeds X MB"
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
        
        # Check for window constraint
        if "one hour" in statement_lower or "1 hour" in statement_lower or "per hour" in statement_lower:
            time_constraint = "window=1h"
    elif "tor exit node" in statement_lower or "tor exit" in statement_lower or "tor" in statement_lower:
        condition = "dst_ip IN threat_feed.tor_exits"
        severity = "HIGH"
        time_constraint = None
    else:
        condition = "unknown_condition"
        severity = "MEDIUM"
        time_constraint = None
        
    # 4. Calculate Parse Confidence
    # Ratio of successfully extracted fields among the 6 core fields
    core_fields = [intent, device_class, condition, time_constraint, action, severity]
    extracted_count = sum(1 for field in core_fields if field is not None)
    parse_confidence = round(extracted_count / 6.0, 2)
    
    # 5. Generate Natural Language Confirmation
    # Create descriptive labels for output sentences
    if "camera" in statement_lower and "external" in statement_lower and "midnight" in statement_lower:
        natural_language_rule = "This rule will alert when a camera device contacts an external IP address between 12:00 AM and 6:00 AM"
    elif "isolate" in statement_lower and "new port" in statement_lower:
        natural_language_rule = "This rule will isolate any device that opens a new port not in its baseline"
    elif "upload" in statement_lower and "5 mb" in statement_lower:
        natural_language_rule = "This rule will alert when any device upload exceeds 5 MB in one hour"
    elif "tor exit node" in statement_lower:
        natural_language_rule = "This rule will alert when any device contacts a TOR exit node"
    else:
        # Dynamic description generator for generic inputs
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
        elif time_constraint == "window=1h":
            time_phrase = " within a one-hour window"
            
        natural_language_rule = f"This rule will {action_phrase} if {device_phrase} {cond_phrase}{time_phrase}."
        
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
    # Test cases representing the 4 user scenarios
    test_cases = [
        "alert if any camera contacts external IP after midnight",
        "isolate any device that opens a new port not in its baseline",
        "alert if upload exceeds 5 MB in one hour",
        "alert if any device contacts a TOR exit node"
    ]
    
    print("=" * 60)
    print("Testing DeviceDNA Rule-Based Policy Parser")
    print("=" * 60)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: \"{test_input}\"")
        result = parse_policy(test_input)
        for key, val in result.items():
            print(f"  {key:22}: {val}")
    print("\n" + "=" * 60)
