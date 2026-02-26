-- DEVICES
CREATE TABLE IF NOT EXISTS devices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id UUID NOT NULL REFERENCES devices(id),
    severity VARCHAR(10) NOT NULL CHECK (
        severity IN ('critical', 'high', 'medium', 'low')
    ),
    alert_type VARCHAR(30) NOT NULL,
    headline TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    confidence FLOAT8 NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    brief JSONB,
    shap_values JSONB,
    response_action VARCHAR(30),
    response_at TIMESTAMPTZ,
    dismissed_by VARCHAR(100),
    dismissed_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
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
