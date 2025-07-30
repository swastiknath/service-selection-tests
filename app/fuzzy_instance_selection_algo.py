#!/usr/bin/env python3
"""
Fuzzy Preference Relations Algorithm Kubernetes Operator
Implements microservice instance selection using fuzzy logic preferences
# Based on Google Kubernetes Engine Documentation
# @Primary Author: Google Cloud
# @Adapted to this Test Case by Swastik Nath (2025)
"""

import asyncio
import logging
import yaml
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
import kopf
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import json
import time
import os

# Based on Google Kubernetes Engine Documentation
# @Primary Author: Google Cloud
# @Adapted to this Test Case by Swastik Nath (2025)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ServiceInstance:
    """Represents a microservice instance with QoS metrics"""
    name: str
    namespace: str
    cpu_usage: float
    memory_usage: float
    response_time: float
    availability: float
    network_latency: float
    error_rate: float
    load: float
    endpoint: str

@dataclass
class FuzzyPreferences:
    """Fuzzy preference weights for different QoS attributes"""
    performance_weight: float = 0.3
    reliability_weight: float = 0.25
    availability_weight: float = 0.2
    efficiency_weight: float = 0.15
    latency_weight: float = 0.1

class FuzzyPreferenceSelector:
    """Implements fuzzy preference relations for instance selection"""
    
    def __init__(self):
        self.setup_fuzzy_system()
    
    def setup_fuzzy_system(self):
        """Initialize fuzzy control system"""
        # Define input variables
        self.cpu_usage = ctrl.Antecedent(np.arange(0, 101, 1), 'cpu_usage')
        self.memory_usage = ctrl.Antecedent(np.arange(0, 101, 1), 'memory_usage')
        self.response_time = ctrl.Antecedent(np.arange(0, 1001, 1), 'response_time')
        self.availability = ctrl.Antecedent(np.arange(0, 101, 1), 'availability')
        self.network_latency = ctrl.Antecedent(np.arange(0, 501, 1), 'network_latency')
        self.error_rate = ctrl.Antecedent(np.arange(0, 51, 1), 'error_rate')
        
        # Define output variable
        self.preference_score = ctrl.Consequent(np.arange(0, 101, 1), 'preference_score')
        
        # Define membership functions
        self._define_membership_functions()
        
        # Create rules
        self._create_fuzzy_rules()
        
        # Create control system
        self.preference_ctrl = ctrl.ControlSystem(self.rules)
        self.preference_sim = ctrl.ControlSystemSimulation(self.preference_ctrl)
    
    def _define_membership_functions(self):
        """Define fuzzy membership functions for all variables"""
        # CPU Usage
        self.cpu_usage['low'] = fuzz.trimf(self.cpu_usage.universe, [0, 0, 30])
        self.cpu_usage['medium'] = fuzz.trimf(self.cpu_usage.universe, [20, 50, 80])
        self.cpu_usage['high'] = fuzz.trimf(self.cpu_usage.universe, [70, 100, 100])
        
        # Memory Usage
        self.memory_usage['low'] = fuzz.trimf(self.memory_usage.universe, [0, 0, 30])
        self.memory_usage['medium'] = fuzz.trimf(self.memory_usage.universe, [20, 50, 80])
        self.memory_usage['high'] = fuzz.trimf(self.memory_usage.universe, [70, 100, 100])
        
        # Response Time
        self.response_time['fast'] = fuzz.trimf(self.response_time.universe, [0, 0, 100])
        self.response_time['medium'] = fuzz.trimf(self.response_time.universe, [50, 200, 400])
        self.response_time['slow'] = fuzz.trimf(self.response_time.universe, [300, 1000, 1000])
        
        # Availability
        self.availability['low'] = fuzz.trimf(self.availability.universe, [0, 0, 70])
        self.availability['medium'] = fuzz.trimf(self.availability.universe, [60, 80, 95])
        self.availability['high'] = fuzz.trimf(self.availability.universe, [90, 100, 100])
        
        # Network Latency
        self.network_latency['low'] = fuzz.trimf(self.network_latency.universe, [0, 0, 50])
        self.network_latency['medium'] = fuzz.trimf(self.network_latency.universe, [30, 100, 200])
        self.network_latency['high'] = fuzz.trimf(self.network_latency.universe, [150, 500, 500])
        
        # Error Rate
        self.error_rate['low'] = fuzz.trimf(self.error_rate.universe, [0, 0, 5])
        self.error_rate['medium'] = fuzz.trimf(self.error_rate.universe, [3, 10, 20])
        self.error_rate['high'] = fuzz.trimf(self.error_rate.universe, [15, 50, 50])
        
        # Preference Score
        self.preference_score['poor'] = fuzz.trimf(self.preference_score.universe, [0, 0, 30])
        self.preference_score['fair'] = fuzz.trimf(self.preference_score.universe, [20, 40, 60])
        self.preference_score['good'] = fuzz.trimf(self.preference_score.universe, [50, 70, 85])
        self.preference_score['excellent'] = fuzz.trimf(self.preference_score.universe, [80, 100, 100])
    
    def _create_fuzzy_rules(self):
        """Create fuzzy inference rules"""
        self.rules = [
            # Excellent performance rules
            ctrl.Rule(self.cpu_usage['low'] & self.memory_usage['low'] & 
                     self.response_time['fast'] & self.availability['high'] & 
                     self.network_latency['low'] & self.error_rate['low'], 
                     self.preference_score['excellent']),
            
            # Good performance rules
            ctrl.Rule(self.cpu_usage['low'] & self.memory_usage['medium'] & 
                     self.response_time['fast'] & self.availability['high'] & 
                     self.network_latency['low'], 
                     self.preference_score['good']),
            
            ctrl.Rule(self.cpu_usage['medium'] & self.memory_usage['low'] & 
                     self.response_time['medium'] & self.availability['medium'] & 
                     self.error_rate['low'], 
                     self.preference_score['good']),
            
            # Fair performance rules
            ctrl.Rule(self.cpu_usage['medium'] & self.memory_usage['medium'] & 
                     self.response_time['medium'] & self.availability['medium'], 
                     self.preference_score['fair']),
            
            ctrl.Rule(self.cpu_usage['low'] & self.response_time['slow'] & 
                     self.error_rate['medium'], 
                     self.preference_score['fair']),
            
            # Poor performance rules
            ctrl.Rule(self.cpu_usage['high'] & self.memory_usage['high'], 
                     self.preference_score['poor']),
            
            ctrl.Rule(self.response_time['slow'] & self.availability['low'] & 
                     self.error_rate['high'], 
                     self.preference_score['poor']),
            
            ctrl.Rule(self.network_latency['high'] & self.error_rate['high'], 
                     self.preference_score['poor']),
        ]
    
    def calculate_preference(self, instance: ServiceInstance) -> float:
        """Calculate fuzzy preference score for an instance"""
        try:
            # Set inputs
            self.preference_sim.input['cpu_usage'] = min(100, max(0, instance.cpu_usage))
            self.preference_sim.input['memory_usage'] = min(100, max(0, instance.memory_usage))
            self.preference_sim.input['response_time'] = min(1000, max(0, instance.response_time))
            self.preference_sim.input['availability'] = min(100, max(0, instance.availability))
            self.preference_sim.input['network_latency'] = min(500, max(0, instance.network_latency))
            self.preference_sim.input['error_rate'] = min(50, max(0, instance.error_rate))
            
            # Compute result
            self.preference_sim.compute()
            
            return self.preference_sim.output['preference_score']
        except Exception as e:
            logger.error(f"Error calculating preference for {instance.name}: {e}")
            return 0.0
    
    def select_best_instance(self, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """Select the best instance based on fuzzy preferences"""
        if not instances:
            return None
        
        best_instance = None
        best_score = -1
        
        for instance in instances:
            score = self.calculate_preference(instance)
            logger.info(f"Instance {instance.name}: score={score:.2f}")
            
            if score > best_score:
                best_score = score
                best_instance = instance
        
        return best_instance

class FuzzyOperator:
    """Kubernetes operator for fuzzy preference-based instance selection"""
    
    def __init__(self):
        self.selector = FuzzyPreferenceSelector()
        self.k8s_client = None
        self.setup_kubernetes()
    
    def setup_kubernetes(self):
        """Initialize Kubernetes client"""
        try:
            # Try in-cluster config first
            config.load_incluster_config()
        except:
            # Fall back to local config
            config.load_kube_config()
        
        self.k8s_client = client.ApiClient()
    
    async def get_service_instances(self, service_name: str, namespace: str) -> List[ServiceInstance]:
        """Get all instances of a service with their metrics"""
        v1 = client.CoreV1Api(self.k8s_client)
        apps_v1 = client.AppsV1Api(self.k8s_client)
        
        instances = []
        
        try:
            # Get service endpoints
            endpoints = v1.read_namespaced_endpoints(service_name, namespace)
            
            # Get deployment for the service
            deployments = apps_v1.list_namespaced_deployment(
                namespace, 
                label_selector=f"app={service_name}"
            )
            
            if not deployments.items:
                return instances
            
            deployment = deployments.items[0]
            
            # Get pods for the deployment
            pods = v1.list_namespaced_pod(
                namespace,
                label_selector=f"app={service_name}"
            )
            
            for pod in pods.items:
                if pod.status.phase == "Running":
                    # Get metrics for the pod (simulated for demo)
                    metrics = await self._get_pod_metrics(pod)
                    
                    instance = ServiceInstance(
                        name=pod.metadata.name,
                        namespace=namespace,
                        cpu_usage=metrics['cpu_usage'],
                        memory_usage=metrics['memory_usage'],
                        response_time=metrics['response_time'],
                        availability=metrics['availability'],
                        network_latency=metrics['network_latency'],
                        error_rate=metrics['error_rate'],
                        load=metrics['load'],
                        endpoint=f"{pod.status.pod_ip}:8080"
                    )
                    instances.append(instance)
            
        except ApiException as e:
            logger.error(f"Error getting service instances: {e}")
        
        return instances
    
    async def _get_pod_metrics(self, pod) -> Dict:
        """Get metrics for a pod (simulated for demo)"""
        # In a real implementation, this would query metrics server or monitoring system
        import random
        
        return {
            'cpu_usage': random.uniform(10, 80),
            'memory_usage': random.uniform(20, 70),
            'response_time': random.uniform(50, 300),
            'availability': random.uniform(85, 99.9),
            'network_latency': random.uniform(10, 100),
            'error_rate': random.uniform(0, 10),
            'load': random.uniform(0, 100)
        }

# Kopf handlers for the operator
@kopf.on.startup()
async def configure(settings: kopf.OperatorSettings, **_):
    """Configure the operator"""
    settings.posting.level = logging.INFO
    logger.info("Fuzzy Preference Relations Operator starting up")

@kopf.on.create('microservice-selector.io', 'v1', 'fuzzyselectors')
async def create_fuzzy_selector(spec, name, namespace, **kwargs):
    """Handle creation of FuzzySelector custom resource"""
    logger.info(f"Creating fuzzy selector: {name}")
    
    operator = FuzzyOperator()
    service_name = spec.get('serviceName')
    
    # Get service instances
    instances = await operator.get_service_instances(service_name, namespace)
    
    if instances:
        # Select best instance
        best_instance = operator.selector.select_best_instance(instances)
        
        if best_instance:
            logger.info(f"Selected best instance: {best_instance.name}")
            
            # Update status
            return {
                'selectedInstance': best_instance.name,
                'selectedEndpoint': best_instance.endpoint,
                'selectionTime': time.time(),
                'algorithm': 'fuzzy-preference-relations'
            }
    
    return {'status': 'no instances available'}

@kopf.on.update('microservice-selector.io', 'v1', 'fuzzyselectors')
async def update_fuzzy_selector(spec, name, namespace, **kwargs):
    """Handle updates to FuzzySelector custom resource"""
    logger.info(f"Updating fuzzy selector: {name}")
    return await create_fuzzy_selector(spec, name, namespace, **kwargs)

@kopf.timer('microservice-selector.io', 'v1', 'fuzzyselectors', interval=30.0)
async def periodic_selection(spec, name, namespace, **kwargs):
    """Periodically re-evaluate instance selection"""
    logger.info(f"Periodic selection for: {name}")
    return await create_fuzzy_selector(spec, name, namespace, **kwargs)

if __name__ == '__main__':
    # Run the operator
    kopf.run()