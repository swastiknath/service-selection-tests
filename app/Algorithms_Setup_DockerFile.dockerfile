# Dockerfile for Fuzzy Preference Relations Operator
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy operator code
COPY fuzzy_instance_selection_algo.py .

# Create health check endpoint
RUN echo '#!/usr/bin/env python3\n\
    import http.server\n\
    import socketserver\n\
    import threading\n\
    import time\n\
    \n\
    class HealthHandler(http.server.BaseHTTPRequestHandler):\n\
    def do_GET(self):\n\
    if self.path == "/healthz" or self.path == "/readiness":\n\
    self.send_response(200)\n\
    self.send_header("Content-type", "text/plain")\n\
    self.end_headers()\n\
    self.wfile.write(b"OK")\n\
    else:\n\
    self.send_response(404)\n\
    self.end_headers()\n\
    \n\
    def start_health_server():\n\
    with socketserver.TCPServer(("", 8080), HealthHandler) as httpd:\n\
    httpd.serve_forever()\n\
    \n\
    if __name__ == "__main__":\n\
    health_thread = threading.Thread(target=start_health_server, daemon=True)\n\
    health_thread.start()\n\
    time.sleep(1)\n' > health_server.py

# Expose port
EXPOSE 8080

# Start both health server and operator
CMD python health_server.py & python fuzzy_instance_selection_algo.py


# Dockerfile for Stochastic QoS Parameters Operator
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy operator code
COPY stochastic_qos_instance_selection.py .

# Copy health server
COPY health_server.py .

# Expose port
EXPOSE 8080

# Start both health server and operator
CMD python health_server.py & python stochastic_qos_instance_selection.py


# Dockerfile for Adaptive Context-Based Operator
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy operator code
COPY adaptive_conext_selection.py .

# Copy health server
COPY health_server.py .

# Expose port
EXPOSE 8080

# Start both health server and operator
CMD python health_server.py & python adaptive_conext_selection.py