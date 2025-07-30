import os
import logging

logger = logging.getLogger(__name__)


class KubernetesTestRunner:
    """Runner for Kubernetes/GKE specific tests"""
    
    def __init__(self):
        self.kubectl_available = self._check_kubectl()
        
    def _check_kubectl(self) -> bool:
        """Check if kubectl is available"""
        try:
            import subprocess
            result = subprocess.run(['kubectl', 'version', '--client'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def generate_k8s_manifests(self, output_dir: str = './k8s-manifests'):
        """Generate Kubernetes manifests for testing"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Service manifest
        service_manifest = """
apiVersion: v1
kind: Service
metadata:
  name: algorithm-test-service
  labels:
    app: algorithm-test
spec:
  selector:
    app: algorithm-test
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: algorithm-test-deployment
  labels:
    app: algorithm-test
spec:
  replicas: 10
  selector:
    matchLabels:
      app: algorithm-test
  template:
    metadata:
      labels:
        app: algorithm-test
    spec:
      containers:
      - name: test-service
        image: nginx:alpine
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
"""
        
        with open(f"{output_dir}/test-service.yaml", 'w') as f:
            f.write(service_manifest)
        
        # Istio VirtualService
        istio_manifest = """
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: algorithm-test-vs
spec:
  hosts:
  - algorithm-test-service
  http:
  - route:
    - destination:
        host: algorithm-test-service
      weight: 100
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: algorithm-test-dr
spec:
  host: algorithm-test-service
  trafficPolicy:
    circuitBreaker:
      consecutiveErrors: 3
      interval: 30s
      baseEjectionTime: 30s
"""
        
        with open(f"{output_dir}/istio-config.yaml", 'w') as f:
            f.write(istio_manifest)
        
        # Linkerd ServiceProfile
        linkerd_manifest = """
apiVersion: linkerd.io/v1alpha2
kind: ServiceProfile
metadata:
  name: algorithm-test-service.default.svc.cluster.local
  namespace: default
spec:
  routes:
  - name: test-route
    condition:
      method: GET
      pathRegex: "/.*"
    responseClasses:
    - condition:
        status:
          min: 200
          max: 299
      isFailure: false
"""
        
        with open(f"{output_dir}/linkerd-config.yaml", 'w') as f:
            f.write(linkerd_manifest)
        
        logger.info(f"Kubernetes manifests generated in {output_dir}")
