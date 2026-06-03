FROM node:20-alpine

WORKDIR /app

# Install dependencies first (cached layer — only re-runs when package.json changes)
COPY package.json package-lock.json* ./
RUN npm install

# Source code is mounted as a volume at runtime, so no COPY needed
