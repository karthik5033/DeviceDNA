import urllib.request, json
url = 'http://localhost:8086/api/v2/query?org=devicedna_org'
query = 'from(bucket: "telemetry") |> range(start: -24h) |> filter(fn: (r) => r["_measurement"] == "trust_scores")'
req = urllib.request.Request(url, data=query.encode('utf-8'), headers={'Authorization': 'Token super-secret-influx-token-123', 'Content-Type': 'application/vnd.flux'})
try:
    resp = urllib.request.urlopen(req)
    data = resp.read().decode('utf-8')
    print('Length:', len(data))
    print(data[:500])
except Exception as e:
    print(e)
