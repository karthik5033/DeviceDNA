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
);

-- PLATFORM SETTINGS
CREATE TABLE IF NOT EXISTS platform_settings (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_by VARCHAR(100)
);

INSERT INTO platform_settings (key, value) VALUES
    ('response_mode', '"advisory"'),
    ('trust_thresholds', '{"critical": 20, "suspicious": 40, "guarded": 60, "normal": 80}')
ON CONFLICT (key) DO NOTHING;

