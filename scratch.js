
        const SERVER_IP = "127.0.0.1"; 
        const PORT = "8000";
        const BASE_URL = `http://${SERVER_IP}:${PORT}`;

        let attackInterval;
        let rpsInterval;
        let totalRequests = 0;
        let requestsThisSecond = 0;
        let isRunning = false;

        async function sendRequest() {
            if (!isRunning) return;
            try {
                // 1. Health endpoint (Burst anomaly)
                await fetch(`${BASE_URL}/api/health`, { mode: 'no-cors', cache: 'no-store' }).catch(e => {});
                totalRequests++;
                requestsThisSecond++;

                // 2. Scan simulation (Real Port scanning anomaly)
                const randomPort = Math.floor(Math.random() * 1000) + 8000;
                await fetch(`http://${SERVER_IP}:${randomPort}/`, { mode: 'no-cors', cache: 'no-store' }).catch(e => {});
                totalRequests++;
                requestsThisSecond++;

                document.getElementById('totalReqs').innerText = totalRequests;
            } catch (err) {
                document.getElementById('errorLog').innerText = "Send Error: " + err.message;
            }
        }

        async function attackLoop() {
            while (isRunning) {
                await sendRequest();
                await new Promise(r => setTimeout(r, 20)); // Small delay to prevent blocking thread
            }
        }

        function startTraffic() {
            try {
                if (isRunning) return;
                isRunning = true;
                
                document.getElementById('errorLog').innerText = ""; // Clear errors
                document.getElementById('launchBtn').disabled = true;
                document.getElementById('launchBtn').innerText = "ATTACKING...";
                document.getElementById('stopBtn').disabled = false;
                
                totalRequests = 0;
                requestsThisSecond = 0;
                document.getElementById('totalReqs').innerText = "0";
                document.getElementById('reqsPerSec').innerText = "0";
                
                // Start async loop instead of setInterval
                attackLoop();
                
                rpsInterval = setInterval(() => {
                    document.getElementById('reqsPerSec').innerText = requestsThisSecond;
                    requestsThisSecond = 0;
                }, 1000);
            } catch (err) {
                document.getElementById('errorLog').innerText = "Start Error: " + err.message;
            }
        }

        function stopTraffic() {
            try {
                if (!isRunning) return;
                isRunning = false;
                
                clearInterval(rpsInterval);
                
                document.getElementById('launchBtn').disabled = false;
                document.getElementById('launchBtn').innerText = "LAUNCH ATTACK";
                document.getElementById('stopBtn').disabled = true;
                
                document.getElementById('reqsPerSec').innerText = "0";
            } catch (err) {
                document.getElementById('errorLog').innerText = "Stop Error: " + err.message;
            }
        }
    