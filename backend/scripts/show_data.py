"""
DeviceDNA — Data Viewer
Shows: PostgreSQL tables (alerts, audit logs, policy rules) + training data summary
Run from backend/ folder with venv active:
    python -m scripts.show_data
"""

import asyncio
import os
import sys
import csv
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SEP = "=" * 90
ROW_SEP = "-" * 90

def hr(title=""):
    if title:
        print(f"\n{SEP}")
        print(f"  {title.upper()}")
        print(SEP)
    else:
        print(ROW_SEP)

def table(headers, rows, col_widths):
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for r in rows:
        truncated = [str(v)[:w] for v, w in zip(r, col_widths)]
        print(fmt.format(*truncated))


# ─── 1. PostgreSQL Data ───────────────────────────────────────────────────────

async def show_postgres():
    try:
        from sqlalchemy import select, func
        from app.db.postgres import AsyncSessionLocal, engine
        from app.db.models import Alert, ResponseAuditLog, PolicyRule

        async with AsyncSessionLocal() as session:

            # ── Alerts ────────────────────────────────────────────────────────
            hr("PostgreSQL — Alerts Table")
            count_res = await session.execute(select(func.count()).select_from(Alert))
            total = count_res.scalar()
            print(f"  Total rows: {total}\n")

            result = await session.execute(
                select(Alert).order_by(Alert.timestamp.desc()).limit(15)
            )
            alerts = result.scalars().all()

            if alerts:
                headers = ["Device ID", "Severity", "Trust Score", "Alert Type", "Resolved", "Timestamp"]
                col_w   = [14,         10,         12,            20,            9,           22]
                rows = [
                    [
                        a.device_id,
                        a.severity.upper(),
                        f"{a.trust_score:.2f}",
                        a.alert_type,
                        "YES" if a.is_resolved else "NO",
                        a.timestamp.strftime("%Y-%m-%d %H:%M:%S") if a.timestamp else "N/A"
                    ]
                    for a in alerts
                ]
                table(headers, rows, col_w)

                # Severity breakdown
                sev_res = await session.execute(
                    select(Alert.severity, func.count()).group_by(Alert.severity)
                )
                sev_counts = sev_res.all()
                print(f"\n  Severity breakdown: " + "  |  ".join(f"{s.upper()}: {c}" for s, c in sev_counts))
            else:
                print("  No alerts found.")

            # ── Response Audit Logs ───────────────────────────────────────────
            hr("PostgreSQL — Response Audit Logs Table")
            count_res = await session.execute(select(func.count()).select_from(ResponseAuditLog))
            total = count_res.scalar()
            print(f"  Total rows: {total}\n")

            result = await session.execute(
                select(ResponseAuditLog).order_by(ResponseAuditLog.timestamp.desc()).limit(15)
            )
            logs = result.scalars().all()

            if logs:
                headers = ["Device ID", "Trigger Score", "Tier", "Action", "HITL Decision", "Timestamp"]
                col_w   = [14,          14,              6,      15,       16,               22]
                rows = [
                    [
                        l.device_id,
                        f"{l.trigger_score:.2f}",
                        f"Tier {l.response_tier}",
                        l.action,
                        l.hitl_decision,
                        l.timestamp.strftime("%Y-%m-%d %H:%M:%S") if l.timestamp else "N/A"
                    ]
                    for l in logs
                ]
                table(headers, rows, col_w)
            else:
                print("  No response audit logs found.")

            # ── Policy Rules ──────────────────────────────────────────────────
            hr("PostgreSQL — Policy Rules Table")
            count_res = await session.execute(select(func.count()).select_from(PolicyRule))
            total = count_res.scalar()
            print(f"  Total rows: {total}\n")

            result = await session.execute(
                select(PolicyRule).order_by(PolicyRule.timestamp.desc()).limit(15)
            )
            rules = result.scalars().all()

            if rules:
                headers = ["Device Class", "Condition", "Action", "Severity", "Active", "Confidence"]
                col_w   = [14,             35,          10,       10,         7,        10]
                rows = [
                    [
                        r.device_class,
                        r.condition,
                        r.action,
                        r.severity,
                        "YES" if r.is_active else "NO",
                        f"{r.parse_confidence:.2f}" if r.parse_confidence else "N/A"
                    ]
                    for r in rules
                ]
                table(headers, rows, col_w)
            else:
                print("  No policy rules found.")

        await engine.dispose()

    except Exception as e:
        print(f"\n  ⚠️  Could not connect to PostgreSQL: {e}")
        print("  Make sure Docker is running and the backend services are up.")
        print("  Run: docker-compose up -d")


# ─── 2. Training Data ─────────────────────────────────────────────────────────

def show_training_data():
    hr("Training Data — physical_devices_baseline.csv")

    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'physical_devices_baseline.csv')
    csv_path = os.path.abspath(csv_path)

    if not os.path.exists(csv_path):
        print(f"  File not found: {csv_path}")
        return

    rows_by_device = defaultdict(list)
    all_rows = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        for row in reader:
            rows_by_device[row['device_id']].append(row)
            all_rows.append(row)

    total_rows = len(all_rows)
    devices = list(rows_by_device.keys())

    print(f"  File:        {csv_path}")
    print(f"  Total rows:  {total_rows}")
    print(f"  Devices:     {len(devices)} — {', '.join(devices)}")
    print(f"  Columns ({len(columns)}): {', '.join(columns)}\n")

    # Per-device summary
    print("  Per-Device Summary:")
    hdr  = ["Device ID",   "Rows", "Date Range (from → to)",                   "Avg Flows", "Avg Bytes",  "Avg Ext Ratio"]
    col_w = [12,            6,      45,                                          10,          12,           14]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*hdr))
    print("  ".join("-" * w for w in col_w))

    for dev_id in devices:
        dev_rows = rows_by_device[dev_id]
        timestamps = sorted(r['timestamp'] for r in dev_rows)
        date_range = f"{timestamps[0][:19]} to {timestamps[-1][:19]}"
        avg_flows = sum(float(r['total_flows']) for r in dev_rows) / len(dev_rows)
        avg_bytes = sum(float(r['total_bytes']) for r in dev_rows) / len(dev_rows)
        avg_ext   = sum(float(r['external_ratio']) for r in dev_rows) / len(dev_rows)
        print(fmt.format(
            dev_id,
            str(len(dev_rows)),
            date_range,
            f"{avg_flows:.1f}",
            f"{avg_bytes:.0f}",
            f"{avg_ext:.4f}"
        ))

    # Sample rows (first 5 rows of first device)
    first_dev = devices[0]
    sample = rows_by_device[first_dev][:5]
    print(f"\n  Sample rows — {first_dev} (first 5 readings):")
    key_cols = ['timestamp', 'total_flows', 'total_bytes', 'avg_packet_size', 'tcp_ratio', 'https_ratio', 'mqtt_ratio', 'unique_dst_ips', 'external_ratio']
    hdr2  = key_cols
    col_w2 = [22, 12, 12, 16, 10, 12, 12, 16, 14]
    fmt2 = "  ".join(f"{{:<{w}}}" for w in col_w2)
    print(fmt2.format(*hdr2))
    print("  ".join("-" * w for w in col_w2))
    for r in sample:
        print(fmt2.format(*[r.get(k, 'N/A')[:w] for k, w in zip(key_cols, col_w2)]))

    # Show trained model files
    hr("Trained Model Files (backend/models_trained/)")
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models_trained'))
    if os.path.exists(model_dir):
        files = sorted(os.listdir(model_dir))
        pt_files  = [f for f in files if f.endswith('.pt')]
        json_files = [f for f in files if f.endswith('.json')]
        joblib_files = [f for f in files if f.endswith('.joblib')]

        print(f"  Total model files: {len(files)}")
        print(f"  PyTorch (.pt):     {len(pt_files)}")
        print(f"  Norm stats (.json):{len(json_files)}")
        print(f"  Isolation Forest:  {len(joblib_files)}\n")

        print("  Model breakdown:")
        groups = {
            "GMVAE Global":     [f for f in pt_files if 'gmvae_global' in f],
            "GMVAE Specialist": [f for f in pt_files if 'gmvae_specialist' in f],
            "LSTM":             [f for f in pt_files if 'lstm' in f],
            "GNN":              [f for f in pt_files if 'gnn' in f],
            "VAE per-device":   [f for f in pt_files if f.startswith('vae_')],
            "Isolation Forest": joblib_files,
        }
        for name, flist in groups.items():
            print(f"    {name:<22}: {len(flist)} files  — {', '.join(flist[:3])}{'...' if len(flist) > 3 else ''}")
    else:
        print("  models_trained/ directory not found.")


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    # Show training data first (no Docker needed)
    show_training_data()

    # Then try PostgreSQL
    hr("PostgreSQL Live Data")
    await show_postgres()

    hr("Done")
    print()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
