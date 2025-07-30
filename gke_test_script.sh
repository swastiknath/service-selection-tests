#!/bin/bash
# GKE Deployment Setup Script for Microservice Algorithm Testing
# Compatible with Istio, Istio Ambient, and Linkerd

# Author: Swastik N. (2025)

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-################}"
CLUSTER_NAME="${CLUSTER_NAME:-algorithm-test-cluster}"
ZONE="${ZONE:-us-central1-a}"
REGION="${REGION:-us-central1}"
MESH_TYPE="${MESH_TYPE:-istio}"  # Options: istio, istio-ambient, linkerd
NODE_COUNT="${NODE_COUNT:-3}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-4}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo_info "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        echo_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        echo_error "kubectl is not installed. Please install it first."
        exit 1
    fi
    
    # Check if helm is installed (for service mesh installation)
    if ! command -v helm &> /dev/null; then
        echo_warning "helm is not installed. It may be needed for service mesh setup."
    fi
    
    echo_success "Prerequisites check completed"
}

# Setup GCP project and authentication
setup_gcp() {
    echo_info "Setting up GCP project and authentication..."
    
    # Set the project
    gcloud config set project $PROJECT_ID
    
    # Enable required APIs
    echo_info "Enabling required GCP APIs..."
    gcloud services enable container.googleapis.com
    gcloud services enable compute.googleapis.com
    gcloud services enable monitoring.googleapis.com
    gcloud services enable logging.googleapis.com
    gcloud services enable cloudtrace.googleapis.com
    
    echo_success "GCP setup completed"
}

# Create GKE cluster
create_cluster() {
    echo_info "Creating GKE cluster: $CLUSTER_NAME"
    
    # Check if cluster already exists
    if gcloud container clusters describe $CLUSTER_NAME --zone=$ZONE &> /dev/null; then
        echo_warning "Cluster $CLUSTER_NAME already exists. Skipping creation."
        return 0
    fi
    
    # Create the cluster with appropriate configuration for service mesh
    gcloud container clusters create $CLUSTER_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --num-nodes=$NODE_COUNT \
        --enable-autoscaling \
        --min-nodes=1 \
        --max-nodes=10 \
        --enable-autorepair \
        --enable-autoupgrade \
        --enable-ip-alias \
        --network="default" \
        --subnetwork="default" \
        --enable-stackdriver-kubernetes \
        --enable-shielded-nodes \
        --workload-pool=${PROJECT_ID}.svc.id.goog \
        --addons HorizontalPodAutoscaling,HttpLoadBalancing,NetworkPolicy
    
    # Get cluster credentials
    gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE
    
    echo_success "GKE cluster created successfully"
}

# Install Istio service mesh
install_istio() {
    echo_info "Installing Istio service mesh..."
    
    # Download and install Istio
    curl -L https://istio.io/downloadIstio | sh -
    export PATH="$PATH:$PWD/istio-*/bin"
    
    # Install Istio with the demo configuration profile
    if [ "$MESH_TYPE" = "istio-ambient" ]; then
        echo_info "Installing Istio in Ambient mode..."
        istioctl install --set values.pilot.env.EXTERNAL_ISTIOD=false \
                        --set values.gateways.istio-ingressgateway.injectionTemplate=gateway \
                        --set values.ztunnel.enabled=true \
                        --set values.pilot.env.ENABLE_WORKLOAD_ENTRY_AUTOREGISTRATION=true \
                        -y
    else
        echo_info "Installing Istio in Sidecar mode..."
        istioctl install --set values.defaultRevision=default -y
    fi
    
    # Label the default namespace for Istio injection
    kubectl label namespace default istio-injection=enabled --overwrite
    
    # Install Istio addons (Prometheus, Grafana, Jaeger, Kiali)
    kubectl apply -f istio-*/samples/addons/
    
    # Wait for Istio components to be ready
    kubectl wait --for=condition=available --timeout=300s deployment --all -n istio-system
    
    echo_success "Istio installation completed"
}

# Install Linkerd service mesh
install_linkerd() {
    echo_info "Installing Linkerd service mesh..."
    
    # Download and install Linkerd CLI
    curl -sL https://run.linkerd.io/install | sh
    export PATH=$PATH:$HOME/.linkerd2/bin
    
    # Pre-check
    linkerd check --pre
    
    # Install Linkerd CRDs
    linkerd install --crds | kubectl apply -f -
    
    # Install Linkerd control plane
    linkerd install | kubectl apply -f -
    
    # Wait for Linkerd to be ready
    linkerd check
    
    # Install Linkerd viz extension
    linkerd viz install | kubectl apply -f -
    
    # Inject Linkerd into the default namespace
    kubectl annotate namespace default linkerd.io/inject=enabled
    
    echo_success "Linkerd installation completed"
}

# Install monitoring stack
install_monitoring() {
    echo_info "Installing monitoring stack..."
    
    # Create monitoring namespace
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    # Install Prometheus using Helm
    if command -v helm &> /dev/null; then
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo update
        
        helm install prometheus prometheus-community/kube-prometheus-stack \
            --namespace monitoring \
            --set grafana.enabled=true \
            --set grafana.adminPassword=admin123 \
            --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
            --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false
    else
        echo_warning "Helm not available, skipping Prometheus installation"
    fi
    
    echo_success "Monitoring stack installation completed"
}

# Deploy the test application
deploy_test_app() {
    echo_info "Deploying test application..."
    
    # Apply the Kubernetes manifests
    kubectl apply -f k8s-manifests/ || {
        echo_error "Failed to apply Kubernetes manifests"
        echo_info "Make sure the k8s-manifests directory exists with the YAML files"
        exit 1
    }
    
    # Wait for deployments to be ready
    echo_info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment --all -n algorithm-test
    
    echo_success "Test application deployed successfully"
}

# Run performance tests
run_tests() {
    echo_info "Running performance tests..."
    
    # Create a job to run the tests
    kubectl create job algorithm-perf-test-$(date +%s) \
        --from=cronjob/algorithm-performance-test-cron \
        -n algorithm-test
    
    echo_info "Performance test job created. Monitor with:"
    echo "  kubectl logs -f job/algorithm-perf-test-* -n algorithm-test"
}

# Setup port forwarding for local access
setup_port_forwarding() {
    echo_info "Setting up port forwarding for local access..."
    
    # Port forward for load balancer
    kubectl port-forward -n algorithm-test service/algorithm-load-balancer 8080:8080 &
    
    # Port forward for Grafana (if installed)
    if kubectl get service -n monitoring prometheus-grafana &> /dev/null; then
        kubectl port-forward -n monitoring service/prometheus-grafana 3000:80 &
    fi
    
    # Port forward for service mesh dashboards
    case $MESH_TYPE in
        "istio"|"istio-ambient")
            if kubectl get service -n istio-system kiali &> /dev/null; then
                kubectl port-forward -n istio-system service/kiali 20001:20001 &
            fi
            if kubectl get service -n istio-system grafana &> /dev/null; then
                kubectl port-forward -n istio-system service/grafana 3001:3000 &
            fi
            ;;
        "linkerd")
            if kubectl get service -n linkerd-viz web &> /dev/null; then
                kubectl port-forward -n linkerd-viz service/web 8084:8084 &
            fi
            ;;
    esac
    
    echo_success "Port forwarding setup completed"
    echo_info "Access points:"
    echo "  - Load Balancer: http://localhost:8080"
    echo "  - Grafana: http://localhost:3000 (admin/admin123)"
    case $MESH_TYPE in
        "istio"|"istio-ambient")
            echo "  - Kiali: http://localhost:20001"
            echo "  - Istio Grafana: http://localhost:3001"
            ;;
        "linkerd")
            echo "  - Linkerd Dashboard: http://localhost:8084"
            ;;
    esac
}

# Cleanup function
cleanup() {
    echo_info "Cleaning up resources..."
    
    # Stop port forwarding
    pkill -f "kubectl port-forward" || true
    
    if [ "$1" = "full" ]; then
        echo_warning "This will delete the entire cluster. Are you sure? (y/N)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            gcloud container clusters delete $CLUSTER_NAME --zone=$ZONE --quiet
            echo_success "Cluster deleted"
        fi
    else
        # Just delete the test namespace
        kubectl delete namespace algorithm-test --ignore-not-found=true
        echo_success "Test namespace deleted"
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "==================================="
    echo "  Microservice Algorithm Test Setup"
    echo "==================================="
    echo "1. Full Setup (Cluster + Service Mesh + App)"
    echo "2. Setup GKE Cluster Only"
    echo "3. Install Service Mesh ($MESH_TYPE)"
    echo "4. Deploy Test Application"
    echo "5. Run Performance Tests"
    echo "6. Setup Port Forwarding"
    echo "7. Cleanup (Keep Cluster)"
    echo "8. Full Cleanup (Delete Cluster)"
    echo "9. Exit"
    echo "==================================="
}

# Process menu selection
process_menu() {
    read -p "Select an option [1-9]: " choice
    case $choice in
        1)
            check_prerequisites
            setup_gcp
            create_cluster
            case $MESH_TYPE in
                "istio"|"istio-ambient") install_istio ;;
                "linkerd") install_linkerd ;;
            esac
            install_monitoring
            deploy_test_app
            setup_port_forwarding
            ;;
        2)
            check_prerequisites
            setup_gcp
            create_cluster
            ;;
        3)
            case $MESH_TYPE in
                "istio"|"istio-ambient") install_istio ;;
                "linkerd") install_linkerd ;;
            esac
            ;;
        4)
            deploy_test_app
            ;;
        5)
            run_tests
            ;;
        6)
            setup_port_forwarding
            ;;
        7)
            cleanup
            ;;
        8)
            cleanup full
            ;;
        9)
            echo_info "Exiting..."
            exit 0
            ;;
        *)
            echo_error "Invalid option. Please try again."
            ;;
    esac
}

# Signal handlers
trap 'echo_info "Interrupted by user"; cleanup; exit 1' INT TERM

# Main execution
main() {
    echo_info "Starting Microservice Algorithm Test Setup"
    echo_info "Project: $PROJECT_ID"
    echo_info "Cluster: $CLUSTER_NAME"
    echo_info "Zone: $ZONE"
    echo_info "Mesh Type: $MESH_TYPE"
    
    # Check if running in interactive mode
    if [ -t 0 ]; then
        while true; do
            show_menu
            process_menu
            echo ""
            read -p "Press Enter to continue..."
        done
    else
        # Non-interactive mode - run full setup
        echo_info "Running in non-interactive mode - performing full setup"
        check_prerequisites
        setup_gcp
        create_cluster
        case $MESH_TYPE in
            "istio"|"istio-ambient") install_istio ;;
            "linkerd") install_linkerd ;;
        esac
        install_monitoring
        deploy_test_app
        run_tests
    fi
}
