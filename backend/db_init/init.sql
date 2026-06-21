
-- DEVICES (Retained for schema completeness, updated device_id/id to VARCHAR for compatibility)
CREATE TABLE IF NOT EXISTS devices (
    id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    mac_address VARCHAR(17) UNIQUE NOT NULL,
    ip_address INET NOT NULL,
    device_class VARCHAR(30) NOT NULL CHECK (
        device_class IN ('camera', 'sensor', 'thermostat', 
                          'access_control', 'medical', 'industrial')
    ),
    vlan INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'online' CHECK (
        status IN ('online', 'offline', 'sandboxed', 'isolated', 'quarantined')
    ),
    firmware_version VARCHAR(50),
    manufacturer VARCHAR(100),
    baseline_complete BOOLEAN DEFAULT FALSE,
    baseline_started_at TIMESTAMPTZ,
    baseline_completed_at TIMESTAMPTZ,
    enrolled_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ALERTS
CREATE TABLE IF NOT EXISTS alerts (
    id VARCHAR(100) PRIMARY KEY,
    device_id VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    trust_score FLOAT NOT NULL,
    vae_score FLOAT NOT NULL,
    if_score FLOAT NOT NULL,
    lstm_score FLOAT NOT NULL,
    gnn_score FLOAT NOT NULL,
    tib JSONB,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_resolved BOOLEAN DEFAULT FALSE
);

-- RESPONSE AUDIT LOGS
CREATE TABLE IF NOT EXISTS response_audit_logs (
    id VARCHAR(100) PRIMARY KEY,
    device_id VARCHAR(100) NOT NULL,
    trigger_score FLOAT NOT NULL,
    response_tier INTEGER NOT NULL,
    action VARCHAR(50) NOT NULL,
    hitl_decision VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP

-- ALERTS Table
CREATE TABLE IF NOT EXISTS alerts (
    id VARCHAR(100) PRIMARY KEY,
    device_id VARCHAR(100) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    alert_type VARCHAR(100) NOT NULL,
    message VARCHAR(255) NOT NULL,
    trust_score FLOAT8 NOT NULL,
    vae_score FLOAT8 NOT NULL,
    if_score FLOAT8 NOT NULL,
    lstm_score FLOAT8 NOT NULL,
    gnn_score FLOAT8 NOT NULL,
    tib JSONB,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    is_resolved BOOLEAN DEFAULT FALSE

);

-- RESPONSE AUDIT LOGS Table
CREATE TABLE IF NOT EXISTS response_audit_logs (
    id VARCHAR(100) PRIMARY KEY,
    device_id VARCHAR(100) NOT NULL,
    trigger_score FLOAT8 NOT NULL,
    response_tier INTEGER NOT NULL,
    action VARCHAR(100) NOT NULL,
    hitl_decision VARCHAR(50) NOT NULL,
    notes TEXT,
    shap_evidence JSONB,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);


INSERT INTO platform_settings (key, value) VALUES
    ('response_mode', '"advisory"'),
    ('trust_thresholds', '{"critical": 20, "suspicious": 40, "guarded": 60, "normal": 80}')
ON CONFLICT (key) DO NOTHING;


-- POLICY RULES Table
CREATE TABLE IF NOT EXISTS policy_rules (
    id VARCHAR(100) PRIMARY KEY,
    device_class VARCHAR(100) NOT NULL DEFAULT 'any',
    condition VARCHAR(500) NOT NULL,
    time_constraint VARCHAR(200),
    action VARCHAR(100) NOT NULL DEFAULT 'alert',
    severity VARCHAR(50) NOT NULL DEFAULT 'MEDIUM',
    natural_language_rule TEXT,
    parse_confidence FLOAT8 DEFAULT 1.0,
    is_active BOOLEAN DEFAULT TRUE,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_alerts_device_id ON alerts(device_id);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_device_id ON response_audit_logs(device_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON response_audit_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_policy_rules_active ON policy_rules(is_active, device_class);

