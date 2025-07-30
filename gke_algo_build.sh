#!/bin/bash
# build-and-deploy.sh - Main build and deployment script

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"your-gcp-project-id"}
REGION=${REGION:-"us-central1"}
CLUSTER_NAME=${CLUSTER_NAME:-"microservice-selector-cluster"}
REGISTRY=${REGISTRY:-"gcr.io/${PROJECT_ID}"}

echo "=== Building and Deploying Microservice Instance Selection Operators ==="
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Cluster: $CLUSTER_NAME"
echo "Registry: $REGISTRY"

# Function to build and push Docker image
build_and_push() {
    local operator_name=$1
    local dockerfile=$2

    echo "Building $operator_name operator..."
    
    # Build Docker image
    docker build -f $dockerfile -t $REGISTRY/microservice-selector/$operator_name:latest .
    
    # Push to registry
    docker push $REGISTRY/microservice-selector/$operator_name:latest
    
    echo "$operator_name operator built and pushed successfully"
}

# Create GKE cluster if it doesn't exist
create_cluster() {
    echo "Checking if cluster exists..."
    
    if ! gcloud container clusters describe $CLUSTER_NAME --region=$REGION &>/dev/null; then
        echo "Creating GKE cluster..."
        gcloud container clusters create $CLUSTER_NAME \
            --region=$REGION \
            --node-locations=$REGION-a,$REGION-b,$REGION-c \
            --num-nodes=2 \
            --enable-autoscaling \
            --min-nodes=1 \
            --max-nodes=5 \
            --enable-autorepair \
            --enable-autoupgrade \
            --machine-type=e2-standard-2 \
            --disk-size=50GB \
            --enable-ip-alias \
            --enable-stackdriver-kubernetes \
            --enable-network-policy
    else
        echo "Cluster already exists"
    fi
    
    # Get cluster credentials
    gcloud container clusters get-credentials $CLUSTER_NAME --region=$REGION
}

# Build health server helper
cat > health_server.py << 'EOF'
#!/usr/bin/env python3
import http.server
import socketserver
import threading
import time

class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz" or self.path == "/readiness":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

def start_health_server():
    with socketserver.TCPServer(("", 8080), HealthHandler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    time.sleep(1)
EOF

# Create Dockerfiles
cat > Dockerfile.fuzzy << 'EOF'
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fuzzy_instance_selection_algo.py .
COPY health_server.py .

EXPOSE 8080

CMD python health_server.py & python fuzzy_instance_selection_algo.py
EOF

cat > Dockerfile.stochastic << 'EOF'
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stochastic_qos_instance_selection.py .
COPY health_server.py .

EXPOSE 8080

CMD python health_server.py & python stochastic_qos_instance_selection.py
EOF

cat > Dockerfile.adaptive << 'EOF'
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY adaptive_context_selection.py .
COPY health_server.py .

EXPOSE 8080

CMD python health_server.py & python adaptive_context_selection.py
EOF

# Main execution
main() {
    echo "Step 1: Creating GKE cluster..."
    create_cluster
    
    echo "Step 2: Building and pushing Docker images..."
    build_and_push "fuzzy-operator" "Dockerfile.fuzzy"
    build_and_push "stochastic-operator" "Dockerfile.stochastic"
    build_and_push "adaptive-operator" "Dockerfile.adaptive"
    
    echo "Step 3: Updating deployment manifests with image URLs..."
    sed -i "s|microservice-selector/fuzzy-operator:latest|$REGISTRY/microservice-selector/fuzzy-operator:latest|g" deployment-manifests.yaml
    sed -i "s|microservice-selector/stochastic-operator:latest|$REGISTRY/microservice-selector/stochastic-operator:latest|g" deployment-manifests.yaml
    sed -i "s|microservice-selector/adaptive-operator:latest|$REGISTRY/microservice-selector/adaptive-operator:latest|g" deployment-manifests.yaml
    
    echo "Step 4: Deploying operators to Kubernetes..."
    kubectl apply -f deployment-manifests.yaml
    
    echo "Step 5: Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/fuzzy-operator
    kubectl wait --for=condition=available --timeout=300s deployment/stochastic-operator
    kubectl wait --for=condition=available --timeout=300s deployment/adaptive-operator
    
    echo "Step 6: Verifying deployments..."
    kubectl get pods -l app=fuzzy-operator
    kubectl get pods -l app=stochastic-operator
    kubectl get pods -l app=adaptive-operator
    
    echo "=== Deployment completed successfully! ==="
    echo ""
    echo "To create example selectors, run:"
    echo "kubectl apply -f example-selectors.yaml"
    echo ""
    echo "To monitor the operators:"
    echo "kubectl logs -f deployment/fuzzy-operator"
    echo "kubectl logs -f deployment/stochastic-operator"
    echo "kubectl logs -f deployment/adaptive-operator"
}

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    if ! command -v gcloud &> /dev/null; then
        echo "gcloud CLI is required but not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        echo "kubectl is required but not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        echo "Docker is required but not installed. Please install it first."
        exit 1
    fi
    
    echo "Prerequisites check passed"
}

# Cleanup function
cleanup() {
    echo "Cleaning up resources..."
    kubectl delete -f deployment-manifests.yaml --ignore-not-found=true
    
    # Optionally delete cluster (uncomment if needed)
    # gcloud container clusters delete $CLUSTER_NAME --region=$REGION --quiet
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        check_prerequisites
        main
        ;;
    "cleanup")
        cleanup
        ;;
    "build-only")
        check_prerequisites
        build_and_push "fuzzy-operator" "Dockerfile.fuzzy"
        build_and_push "stochastic-operator" "Dockerfile.stochastic"
        build_and_push "adaptive-operator" "Dockerfile.adaptive"
        ;;
    *)
        echo "Usage: $0 [deploy|cleanup|build-only]"
        echo "  deploy     - Build images and deploy to GKE (default)"
        echo "  cleanup    - Remove deployed resources"
        echo "  build-only - Only build and push Docker images"
        exit 1
        ;;
esac