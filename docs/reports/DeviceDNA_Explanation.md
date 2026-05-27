# DeviceDNA: A Simple Guide

## 1. What is this project?
**DeviceDNA** is an AI-powered cybersecurity platform built specifically for the Internet of Things (IoT). Imagine a massive network with thousands of smart devices—like smart cameras, thermostats, medical sensors, and factory controllers. Because these devices often lack traditional security software and run constantly, hackers can easily compromise them and remain undetected for months.

DeviceDNA constantly monitors the network traffic of every single IoT device. Instead of relying on a hard-coded list of known viruses or firewall rules, DeviceDNA uses Artificial Intelligence to learn how each device *usually* behaves (creating what's called a **Digital Twin**). It calculates a real-time **Trust Score** (0 to 100) for every device based on its behavior. The moment a device starts acting strangely—such as a smart thermostat trying to connect to a foreign server, or a camera sending ten times more data than usual—its Trust Score drops, alerts the user, and can even automatically isolate the infected device.

## 2. Architecture: How is it built?
The project consists of four main layers:

1. **Data Infrastructure (The Storage & Pipelines)**: This layer handles massive amounts of device data continuously. It includes Kafka, Redis, InfluxDB, and PostgreSQL, all running inside Docker containers.
2. **Telemetry Simulator (The Fake Traffic Generator)**: Since we can't test this directly on a real hospital or factory network safely, a Python script fakes a network of 50 devices. It simulates normal, healthy traffic and also actively generates complex "attacks" (like botnets, data theft, or stealthy data leaks) for our system to catch.
3. **ML Backend API (The Brains)**: A Python application built with **FastAPI** that runs all the Machine Learning (ML) models to evaluate incoming traffic, calculate trust scores, and identify anomalies.
4. **SOC Dashboard (The User Interface)**: A beautiful web dashboard built with **Next.js** and **React** (using visual charting tools like D3.js and Recharts). It acts as the command center, displaying live trust scores, highlighting alerts, and showing an interactive spider-web map of how devices are communicating.

## 3. Why are we using specific technologies?

### Docker & Docker Compose
Think of Docker like shipping containers for software. Instead of having to download and install a database, a cache server, a message broker, and a specific Python environment manually step-by-step—which is extremely prone to crashing and errors across different computers—Docker packages all of them into neat "containers". You simply run `docker compose up`, and the entire complex data setup magically boots up exactly as intended.

### Apache Kafka (with Zookeeper)
In a real IoT environment, thousands of devices are sending network packets every single millisecond. Normal databases would lag, panic, and crash under this massive flood of data. Kafka acts as a super-fast, incredibly durable **message queue**. It safely catches all these incoming high-speed data events, lines them up, and smoothly feeds them to our Machine Learning processing engine at a pace the engine can digest.

### Redis
Redis is an **in-memory caching database**. Because it stores data directly in RAM (memory) instead of saving it to a slower hard drive, it is lightning-fast. DeviceDNA uses Redis to quickly fetch, calculate, and update the live, running trust scores for the dashboard in real-time. 

### PostgreSQL & InfluxDB
- **PostgreSQL**: A traditional relational database. We use this to safely store user settings, configuration rules, login accounts, and predefined security policies.
- **InfluxDB**: A specialized "**Time-Series**" database. Network data is fundamentally just a sequence of events happening over time. InfluxDB is specifically optimized to quickly store and search massive amounts of data grouped by timestamps, making it perfect for historical behavior tracking.

## 4. Why Machine Learning (ML) Models?
Hackers today are incredibly smart. They don't typically trip traditional alarm bells by aggressively crashing systems. Instead, they execute "stealth attacks" (like stealing a few kilobytes of data every single day). Ordinary rules-based firewalls (e.g., "Block Port 22") fail entirely because the hacker can just use allowed ports and normal traffic limits.

Machine Learning is required because it's the only way to learn subtle patterns of "normalcy" and catch strange new variations automatically. 

Here are the specific ML models used within DeviceDNA and why:

- **Autoencoders (Digital Twin)**: This model looks at a device's first 7 days of life and learns its exact mathematical behavior footprint (its "Digital Twin"). Once trained, if a hacker takes over a smart camera and makes it act even slightly differently, the Autoencoder notices the reality no longer matches the historical Twin, and raises an alarm.
- **Isolation Forests**: This model is fantastic at spotting weird outliers in massive datasets. If a thermostat suddenly uses an unusual file transfer protocol once out of 500,000 normal events, the Isolation Forest flags that specific action as an anomaly.
- **LSTM (Long Short-Term Memory) & CUSUM**: These models handle the sneaky attacks known as **Soft Drift**. If an attacker increases the camera's data upload by just 1% each day, standard alarms won't notice at first. LSTM and CUSUM track long-term memory to realize that, over the span of 14 days, the subtle behavior shift has pushed the device into a dangerous zone.
- **Graph Neural Networks (GNN)**: Devices don't operate alone—they talk to each other. GNNs map the entire network like a spiderweb. If a hacked temperature sensor suddenly starts trying to communicate directly with a central database server (a hacker tactic known as "Lateral Movement"), the GNN immediately spots two components of the web connecting in a way they never usually do.
