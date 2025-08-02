#!/usr/bin/env python3
"""
Stochastic QoS Parameters Algorithm Kubernetes Operator
Implements probabilistic microservice instance selection based on QoS parameters

# Based on Google Kubernetes Engine Documentation
# @Primary Author: Google Cloud
# @Adapted to this Test Case by Swastik Nath (2025)
"""

import asyncio
import logging
import yaml
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
import kopf
import scipy.stats as stats
from scipy.optimize import minimize
import json
import time
import os
from collections import defaultdict, deque
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QoSMetrics:
    """QoS metrics with statistical properties"""
    response_time: float
    throughput: float
    availability: float
    reliability: float
    cpu_utilization: float
    memory_utilization: float
    network_latency: float
    error_rate: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class StochasticInstance:
    """Service instance with stochastic QoS modeling"""
    name: str
    namespace: str
    endpoint: str
    metrics_history: deque = field(default_factory=lambda: deque(maxlen=100))
    qos_distributions: Dict = field(default_factory=dict)
    selection_probability: float = 0.0
    last_updated: float = field(default_factory=time.time)

@dataclass
class QoSRequirements:
    """QoS requirements with stochastic constraints"""
    max_response_time: Tuple[float, float]  # (mean, std_dev)
    min_throughput: Tuple[float, float]
    min_availability: Tuple[float, float]
    max_error_rate: Tuple[float, float]
    weights: Dict[str, float] = field(default_factory=lambda: {
        'response_time': 0.25,
        'throughput': 0.20,
        'availability': 0.20,
        'reliability': 0.15,
        'resource_efficiency': 0.10,
        'network_performance': 0.10
    })

class StochasticQoSSelector:
    """Implements stochastic QoS-based instance selection"""
    
    def __init__(self):
        self.instances = {}
        self.selection_history = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def update_instance_metrics(self, instance_name: str, metrics: QoSMetrics):
        """Update metrics for an instance and recalculate distributions"""
        with self.lock:
            if instance_name not in self.instances:
                self.instances[instance_name] = StochasticInstance(
                    name=instance_name,
                    namespace="default",
                    endpoint=""
                )
            
            instance = self.instances[instance_name]
            instance.metrics_history.append(metrics)
            instance.last_updated = time.time()
            
            # Update statistical distributions
            self._update_distributions(instance)
    
    def _update_distributions(self, instance: StochasticInstance):
        """Update probability distributions for QoS parameters"""
        if len(instance.metrics_history) < 3:
            return
        
        metrics_arrays = {
            'response_time': [],
            'throughput': [],
            'availability': [],
            'reliability': [],
            'cpu_utilization': [],
            'memory_utilization': [],
            'network_latency': [],
            'error_rate': []
        }
        
        # Collect historical data
        for metrics in instance.metrics_history:
            metrics_arrays['response_time'].append(metrics.response_time)
            metrics_arrays['throughput'].append(metrics.throughput)
            metrics_arrays['availability'].append(metrics.availability)
            metrics_arrays['reliability'].append(metrics.reliability)
            metrics_arrays['cpu_utilization'].append(metrics.cpu_utilization)
            metrics_arrays['memory_utilization'].append(metrics.memory_utilization)
            metrics_arrays['network_latency'].append(metrics.network_latency)
            metrics_arrays['error_rate'].append(metrics.error_rate)
        
        # Fit distributions
        instance.qos_distributions = {}
        for param, values in metrics_arrays.items():
            if len(values) >= 3:
                try:
                    # Fit normal distribution
                    mu, sigma = stats.norm.fit(values)
                    instance.qos_distributions[param] = {
                        'type': 'normal',
                        'params': {'mu': mu, 'sigma': sigma},
                        'current': values[-1]
                    }
                except Exception as e:
                    logger.warning(f"Could not fit distribution for {param}: {e}")
    
    def calculate_satisfaction_probability(self, instance: StochasticInstance, 
                                         requirements: QoSRequirements) -> float:
        """Calculate probability that instance satisfies QoS requirements"""
        if not instance.qos_distributions:
            return 0.0
        
        probabilities = []
        
        # Response time probability
        if 'response_time' in instance.qos_distributions:
            dist = instance.qos_distributions['response_time']
            if dist['type'] == 'normal':
                mu, sigma = dist['params']['mu'], dist['params']['sigma']
                req_mean, req_std = requirements.max_response_time
                # Probability that response time is less than requirement
                prob = stats.norm.cdf(req_mean, mu, sigma)
                probabilities.append(('response_time', prob))
        
        # Throughput probability
        if 'throughput' in instance.qos_distributions:
            dist = instance.qos_distributions['throughput']
            if dist['type'] == 'normal':
                mu, sigma = dist['params']['mu'], dist['params']['sigma']
                req_mean, req_std = requirements.min_throughput
                # Probability that throughput is greater than requirement
                prob = 1 - stats.norm.cdf(req_mean, mu, sigma)
                probabilities.append(('throughput', prob))
        
        # Availability probability
        if 'availability' in instance.qos_distributions:
            dist = instance.qos_distributions['availability']
            if dist['type'] == 'normal':
                mu, sigma = dist['params']['mu'], dist['params']['sigma']
                req_mean, req_std = requirements.min_availability
                prob = 1 - stats.norm.cdf(req_mean, mu, sigma)
                probabilities.append(('availability', prob))
        
        # Error rate probability
        if 'error_rate' in instance.qos_distributions:
            dist = instance.qos_distributions['error_rate']
            if dist['type'] == 'normal':
                mu, sigma = dist['params']['mu'], dist['params']['sigma']
                req_mean, req_std = requirements.max_error_rate
                prob = stats.norm.cdf(req_mean, mu, sigma)
                probabilities.append(('error_rate', prob))
        
        if not probabilities:
            return 0.0
        
        # Calculate weighted probability
        total_prob = 0.0
        total_weight = 0.0
        
        for param_name, prob in probabilities:
            weight = requirements.weights.get(param_name, 0.1)
            total_prob += prob * weight
            total_weight += weight
        
        return total_prob / total_weight if total_weight > 0 else 0.0
    
    def calculate_utility_score(self, instance: StochasticInstance,
                               requirements: QoSRequirements) -> float:
        """Calculate utility score considering multiple QoS aspects"""
        if not instance.qos_distributions:
            return 0.0
        
        utility_components = []
        
        # Performance utility (response time, throughput)
        perf_score = 0.0
        if 'response_time' in instance.qos_distributions:
            rt_dist = instance.qos_distributions['response_time']
            if rt_dist['type'] == 'normal':
                # Lower response time is better
                normalized_rt = 1.0 / (1.0 + rt_dist['current'])
                perf_score += normalized_rt * 0.5
        
        if 'throughput' in instance.qos_distributions:
            tp_dist = instance.qos_distributions['throughput']
            if tp_dist['type'] == 'normal':
                # Higher throughput is better
                normalized_tp = tp_dist['current'] / 1000.0  # Assume max 1000 req/s
                perf_score += min(normalized_tp, 1.0) * 0.5
        
        utility_components.append(('performance', perf_score))
        
        # Reliability utility (availability, error rate)
        rel_score = 0.0
        if 'availability' in instance.qos_distributions:
            avail_dist = instance.qos_distributions['availability']
            if avail_dist['type'] == 'normal':
                rel_score += avail_dist['current'] / 100.0 * 0.6
        
        if 'error_rate' in instance.qos_distributions:
            err_dist = instance.qos_distributions['error_rate']
            if err_dist['type'] == 'normal':
                # Lower error rate is better
                rel_score += max(0, 1.0 - err_dist['current'] / 10.0) * 0.4
        
        utility_components.append(('reliability', rel_score))
        
        # Resource efficiency utility
        eff_score = 0.0
        if 'cpu_utilization' in instance.qos_distributions:
            cpu_dist = instance.qos_distributions['cpu_utilization']
            if cpu_dist['type'] == 'normal':
                # Moderate CPU usage is optimal
                cpu_util = cpu_dist['current']
                if 30 <= cpu_util <= 70:
                    eff_score += 1.0 * 0.5
                else:
                    eff_score += max(0, 1.0 - abs(cpu_util - 50) / 50.0) * 0.5
        
        if 'memory_utilization' in instance.qos_distributions:
            mem_dist = instance.qos_distributions['memory_utilization']
            if mem_dist['type'] == 'normal':
                mem_util = mem_dist['current']
                if 30 <= mem_util <= 70:
                    eff_score += 1.0 * 0.5
                else:
                    eff_score += max(0, 1.0 - abs(mem_util - 50) / 50.0) * 0.5
        
        utility_components.append(('efficiency', eff_score))
        
        # Calculate weighted utility
        total_utility = 0.0
        for component, score in utility_components:
            weight = requirements.weights.get(component, 0.33)
            total_utility += score * weight
        
        return total_utility
    
    def select_instances_stochastic(self, requirements: QoSRequirements,
                                  num_instances: int = 1) -> List[StochasticInstance]:
        """Select instances using stochastic optimization"""
        with self.lock:
            if not self.instances:
                return []
            
            instance_list = list(self.instances.values())
            
            # Calculate selection probabilities for each instance
            for instance in instance_list:
                satisfaction_prob = self.calculate_satisfaction_probability(instance, requirements)
                utility_score = self.calculate_utility_score(instance, requirements)
                
                # Combine probability and utility
                instance.selection_probability = satisfaction_prob * 0.6 + utility_score * 0.4
            
            # Sort by selection probability
            instance_list.sort(key=lambda x: x.selection_probability, reverse=True)
            
            # Apply stochastic selection with weighted probabilities
            selected = []
            available_instances = instance_list.copy()
            
            for _ in range(min(num_instances, len(available_instances))):
                if not available_instances:
                    break
                
                # Calculate selection weights (higher probability = higher weight)
                weights = [inst.selection_probability + 0.1 for inst in available_instances]
                total_weight = sum(weights)
                
                if total_weight == 0:
                    selected.append(available_instances[0])
                    available_instances.remove(available_instances[0])
                else:
                    # Weighted random selection
                    probabilities = [w / total_weight for w in weights]
                    selected_idx = np.random.choice(len(available_instances), p=probabilities)
                    selected_instance = available_instances[selected_idx]
                    selected.append(selected_instance)
                    available_instances.remove(selected_instance)
            
            # Update selection history
            self.selection_history.extend(selected)
            
            return selected
    
    def get_selection_statistics(self) -> Dict:
        """Get statistics about instance selections"""
        if not self.selection_history:
            return {}
        
        selection_counts = defaultdict(int)
        for instance in self.selection_history:
            selection_counts[instance.name] += 1
        
        total_selections = len(self.selection_history)
        
        stats_dict = {
            'total_selections': total_selections,
            'instance_selection_rates': {
                name: count / total_selections 
                for name, count in selection_counts.items()
            },
            'active_instances': len(self.instances)
        }
        
        return stats_dict

class StochasticOperator:
    """Kubernetes operator for stochastic QoS-based instance selection"""
    
    def __init__(self):
        self.selector = StochasticQoSSelector()
        self.k8s_client = None
        self.setup_kubernetes()
    
    def setup_kubernetes(self):
        """Initialize Kubernetes client"""
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.k8s_client = client.ApiClient()
    
    async def collect_instance_metrics(self, service_name: str, namespace: str):
        """Collect metrics for all service instances"""
        v1 = client.CoreV1Api(self.k8s_client)
        
        try:
            pods = v1.list_namespaced_pod(
                namespace,
                label_selector=f"app={service_name}"
            )
            
            for pod in pods.items:
                if pod.status.phase == "Running":
                    metrics = await self._simulate_metrics_collection(pod)
                    qos_metrics = QoSMetrics(**metrics)
                    self.selector.update_instance_metrics(pod.metadata.name, qos_metrics)
                    
        except ApiException as e:
            logger.error(f"Error collecting metrics: {e}")
    
    