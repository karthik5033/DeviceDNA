# Pre-Demo Checklist

Run through this checklist 10 minutes before the hackathon demo to ensure the entire DeviceDNA platform is fully operational and primed with data.

### Infrastructure & Backend

1. **[ ] Docker Health Checks**
   - Run `docker-compose ps` to ensure all containers (`postgres`, `influxdb`, `redis`, `kafka`, `zookeeper`, `seeder`, `backend`, `frontend`, `simulator`) are up.
   - Verify DB health checks say `(healthy)`.

2. **[ ] Seeder Verification**
   - Check the seeder logs: `docker-compose logs seeder`
   - You should see `Demo data seeded successfully` and the container should have exited cleanly (Status 0).
   - This guarantees the PostgreSQL and InfluxDB initial state is ready.

3. **[ ] Simulator Validation**
   - Check simulator logs: `docker-compose logs simulator`
   - You should see lines like `Emitting flows for 50 devices...`. This confirms the Kafka pipeline is ingesting real-time traffic.

### Dashboard & UI

4. **[ ] Topology Node Count**
   - Open `http://localhost:3000/dashboard/topology` (or main dashboard).
   - Verify that exactly 50 device nodes are rendered and live-updating on the screen.

5. **[ ] Threat Alert Feed**
   - Navigate to the **Alerts** page (`/dashboard/alerts`).
   - Verify that there is at least one **Critical** alert visible in the feed (triggered by the seeded anomalous data).
   - Click it to ensure the SHAP Explainable AI panel loads on the right.

6. **[ ] NLP Policy Engine**
   - Navigate to the **Policies** page (`/dashboard/policies`).
   - Type a test policy (e.g., "Block all cameras from sending more than 10MB").
   - Ensure the NLP engine responds and parses the rule without errors.

7. **[ ] Trust History Data**
   - Click into any specific Node page (e.g. `SIM-0001`).
   - Verify the line chart (**Trust Score History**) is rendering and contains historical plot points (not empty/loading).

8. **[ ] Autonomous Response Panel**
   - On a node page for a **Critical** device (e.g., one that triggered an alert), look at the bottom left panel.
   - Verify the **Autonomous Response Status** panel is visible.
   - Confirm that the `Forensic` badge is yellow or the `Isolated` badge is red (confirming the response engine fired).
