#!/usr/bin/env python3
"""
Adaptive Context-Based Instance Selection Algorithm Kubernetes Operator
Implements dynamic microservice instance selection based on contextual factors

# Based on Google Kubernetes Engine Documentation
# @Primary Author: Google Cloud
# @Adapted to this Test Case by Swastik Nath (2025)
"""

import asyncio
import logging
import yaml
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
import kopf
import json
import time
import os
from collections import defaultdict, deque
import threading
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContextualFactors:
    """Contextual factors affecting instance selection"""
    timestamp: float
    request_type: str
    user_location: str
    time_of_day: int  # 0-23
    day_of_week: int  # 0-6
    traffic_load: float
    network_conditions: str
    user_priority: str
    session_type: str
    data_sensitivity: str
    geographical_zone: str
    device_type: str

@dataclass
class InstanceContext:
    """Instance with contextual information"""
    name: str
    namespace: str
    endpoint: str
    zone: str
    capabilities: List[str]
    resource_profile: Dict[str, float]
    performance_history: deque = field(default_factory=lambda: deque(maxlen=200))
    context_affinity: Dict[str, float] = field(default_factory=dict)
    adaptation_score: float = 0.0
    last_selected: float = 0.0
    selection_count: int = 0

@dataclass
class PerformanceMetrics:
    """Performance metrics with contextual tags"""
    response_time: float
    throughput: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    network_latency: float
    success_rate: float
    context: ContextualFactors
    timestamp: float = field(default_factory=time.time)

class ContextPattern:
    """Represents a learned context pattern"""
    def __init__(self, pattern_id: str):
        self.pattern_id = pattern_id
        self.context_features = {}
        self.performance_expectations = {}
        self.preferred_instances = []
        self.confidence = 0.0
        self.usage_count = 0
        self.last_updated = time.time()

class AdaptiveContextSelector:
    """Implements adaptive context-based instance selection"""
    
    def __init__(self):
        self.instances = {}
        self.context_patterns = {}
        self.performance_predictor = None
        self.context_clusterer = None
        self.scaler = StandardScaler()
        self.selection_history = deque(maxlen=1000)
        self.adaptation_weights = {
            'historical_performance': 0.3,
            'context_similarity': 0.25,
            'load_balancing': 0.2,
            'geographical_affinity': 0.15,
            'resource_efficiency': 0.1
        }
        self.lock = threading.Lock()
        self._initialize_ml_models()
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for adaptation"""
        self.performance_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.context_clusterer = KMeans(n_clusters=5, random_state=42)
    
    def register_instance(self, instance: InstanceContext):
        """Register a new service instance"""
        with self.lock:
            self.instances[instance.name] = instance
            logger.info(f"Registered instance: {instance.name}")
    
    def update_performance_metrics(self, instance_name: str, metrics: PerformanceMetrics):
        """Update performance metrics for an instance"""
        with self.lock:
            if instance_name in self.instances:
                self.instances[instance_name].performance_history.append(metrics)
                self._update_context_affinity(instance_name, metrics)
                self._learn_context_patterns(metrics)
    
    def _update_context_affinity(self, instance_name: str, metrics: PerformanceMetrics):
        """Update context affinity based on performance"""
        instance = self.instances[instance_name]
        context = metrics.context
        
        # Create context signature
        context_signature = f"{context.request_type}_{context.user_location}_{context.time_of_day}_{context.traffic_load:.1f}"
        
        # Calculate performance score
        perf_score = self._calculate_performance_score(metrics)
        
        # Update affinity with exponential smoothing
        if context_signature in instance.context_affinity:
            instance.context_affinity[context_signature] = (
                0.7 * instance.context_affinity[context_signature] + 
                0.3 * perf_score
            )
        else:
            instance.context_affinity[context_signature] = perf_score
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate normalized performance score"""
        # Normalize metrics (lower is better for response_time, error_rate, latency)
        rt_score = max(0, 1 - metrics.response_time / 1000)  # Assume 1000ms max
        throughput_score = min(1, metrics.throughput / 1000)  # Assume 1000 req/s max
        error_score = max(0, 1 - metrics.error_rate / 10)  # Assume 10% max error
        success_score = metrics.success_rate / 100
        
        # Resource efficiency
        resource_score = max(0, 1 - (metrics.cpu_usage + metrics.memory_usage) / 200)
        
        # Network performance
        network_score = max(0, 1 - metrics.network_latency / 500)  # 500ms max
        
        # Weighted average
        total_score = (
            rt_score * 0.25 +
            throughput_score * 0.2 +
            error_score * 0.2 +
            success_score * 0.15 +
            resource_score * 0.1 +
            network_score * 0.1
        )
        
        return total_score
    
    def _learn_context_patterns(self, metrics: PerformanceMetrics):
        """Learn patterns from contextual performance data"""
        context = metrics.context
        
        # Create pattern key based on context features
        pattern_key = f"{context.request_type}_{context.time_of_day//4}_{context.day_of_week}_{context.traffic_load//0.2}"
        
        if pattern_key not in self.context_patterns:
            self.context_patterns[pattern_key] = ContextPattern(pattern_key)
        
        pattern = self.context_patterns[pattern_key]
        pattern.usage_count += 1
        pattern.last_updated = time.time()
        
        # Update performance expectations
        perf_score = self._calculate_performance_score(metrics)
        if 'average_performance' in pattern.performance_expectations:
            pattern.performance_expectations['average_performance'] = (
                0.8 * pattern.performance_expectations['average_performance'] + 
                0.2 * perf_score
            )
        else:
            pattern.performance_expectations['average_performance'] = perf_score
        
        # Update confidence based on usage
        pattern.confidence = min(1.0, pattern.usage_count / 50.0)
    
    def _extract_context_features(self, context: ContextualFactors) -> np.ndarray:
        """Extract numerical features from context"""
        features = [
            context.time_of_day / 24.0,
            context.day_of_week / 7.0,
            context.traffic_load,
            hash(context.request_type) % 100 / 100.0,
            hash(context.user_location) % 100 / 100.0,
            hash(context.network_conditions) % 10 / 10.0,
            hash(context.user_priority) % 5 / 5.0,
            hash(context.session_type) % 10 / 10.0,
            hash(context.data_sensitivity) % 5 / 5.0,
            hash(context.geographical_zone) % 20 / 20.0,
            hash(context.device_type) % 10 / 10.0
        ]
        return np.array(features)
    
    def _calculate_context_similarity(self, context1: ContextualFactors, 
                                    context2: ContextualFactors) -> float:
        """Calculate similarity between two contexts"""
        features1 = self._extract_context_features(context1)
        features2 = self._extract_context_features(context2)
        
        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norms = np.linalg.norm(features1) * np.linalg.norm(features2)
        
        if norms == 0:
            return 0.0
        
        return dot_product / norms
    
    def _predict_instance_performance(self, instance: InstanceContext, 
                                    context: ContextualFactors) -> float:
        """Predict instance performance for given context"""
        if len(instance.performance_history) < 5:
            return 0.5  # Default score for new instances
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for metrics in instance.performance_history:
            context_features = self._extract_context_features(metrics.context)
            instance_features = [
                instance.resource_profile.get('cpu_capacity', 1.0),
                instance.resource_profile.get('memory_capacity', 1.0),
                instance.resource_profile.get('network_bandwidth', 1.0),
                len(instance.capabilities) / 10.0
            ]
            
            combined_features = np.concatenate([context_features, instance_features])
            X_train.append(combined_features)
            y_train.append(self._calculate_performance_score(metrics))
        
        if len(X_train) < 3:
            return 0.5
        
        try:
            # Train predictor
            self.performance_predictor.fit(X_train, y_train)
            
            # Predict for current context
            current_context_features = self._extract_context_features(context)
            instance_features = [
                instance.resource_profile.get('cpu_capacity', 1.0),
                instance.resource_profile.get('memory_capacity', 1.0),
                instance.resource_profile.get('network_bandwidth', 1.0),
                len(instance.capabilities) / 10.0
            ]
            
            combined_features = np.concatenate([current_context_features, instance_features])
            prediction = self.performance_predictor.predict([combined_features])[0]
            
            return max(0.0, min(1.0, prediction))
        
        except Exception as e:
            logger.warning(f"Prediction failed for {instance.name}: {e}")
            return 0.5
    
    def _calculate_load_balancing_score(self, instance: InstanceContext) -> float:
        """Calculate load balancing score (prefer less loaded instances)"""
        current_time = time.time()
        recent_selections = sum(1 for _ in self.selection_history 
                              if _.name == instance.name and current_time - _.last_selected < 300)
        
        # Inverse relationship with recent selections
        max_recent = max(sum(1 for _ in self.selection_history 
                           if _.name == inst.name and current_time - _.last_selected < 300)
                        for inst in self.instances.values()) or 1
        
        return 1.0 - (recent_selections / max_recent)
    
    def _calculate_geographical_affinity(self, instance: InstanceContext,
                                       context: ContextualFactors) -> float:
        """Calculate geographical affinity score"""
        # Simple zone matching (in production, use actual geographical calculations)
        if hasattr(instance, 'zone') and hasattr(context, 'geographical_zone'):
            if instance.zone == context.geographical_zone:
                return 1.0
            elif instance.zone.split('-')[0] == context.geographical_zone.split('-')[0]:
                return 0.7  # Same region, different zone
            else:
                return 0.3  # Different region
        
        return 0.5  # Default if zone info not available
    
    def select_optimal_instance(self, context: ContextualFactors,
                              requirements: Dict[str, Any] = None) -> Optional[InstanceContext]:
        """Select optimal instance based on adaptive context analysis"""
        with self.lock:
            if not self.instances:
                return None
            
            instance_scores = {}
            
            for instance_name, instance in self.instances.items():
                scores = {}
                
                # Historical performance score
                predicted_perf = self._predict_instance_performance(instance, context)
                scores['historical_performance'] = predicted_perf
                
                # Context similarity score
                context_sim_scores = []
                for metrics in list(instance.performance_history)[-20:]:  # Last 20 records
                    sim = self._calculate_context_similarity(context, metrics.context)
                    perf = self._calculate_performance_score(metrics)
                    context_sim_scores.append(sim * perf)
                
                scores['context_similarity'] = np.mean(context_sim_scores) if context_sim_scores else 0.5
                
                # Load balancing score
                scores['load_balancing'] = self._calculate_load_balancing_score(instance)
                
                # Geographical affinity score
                scores['geographical_affinity'] = self._calculate_geographical_affinity(instance, context)
                
                # Resource efficiency score
                if instance.performance_history:
                    recent_metrics = list(instance.performance_history)[-5:]
                    avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
                    avg_mem = np.mean([m.memory_usage for m in recent_metrics])
                    scores['resource_efficiency'] = max(0, 1 - (avg_cpu + avg_mem) / 200)
                else:
                    scores['resource_efficiency'] = 0.5
                
                # Calculate weighted total score
                total_score = sum(
                    scores[component] * self.adaptation_weights[component]
                    for component in scores
                )
                
                instance_scores[instance_name] = {
                    'total_score': total_score,
                    'component_scores': scores,
                    'instance': instance
                }
            
            # Select instance with highest score
            best_instance_name = max(instance_scores.keys(), 
                                   key=lambda x: instance_scores[x]['total_score'])
            
            best_instance = instance_scores[best_instance_name]['instance']
            best_instance.last_selected = time.time()
            best_instance.selection_count += 1
            
            # Update adaptation score for learning
            best_instance.adaptation_score = instance_scores[best_instance_name]['total_score']
            
            # Log selection details
            logger.info(f"Selected instance {best_instance_name} with score "
                       f"{instance_scores[best_instance_name]['total_score']:.3f}")
            logger.debug(f"Component scores: {instance_scores[best_instance_name]['component_scores']}")
            
            return best_instance
    
    def adapt_selection_weights(self, feedback: Dict[str, float]):
        """Adapt selection weights based on performance feedback"""
        with self.lock:
            # Simple adaptive weight adjustment
            for component, feedback_score in feedback.items():
                if component in self.adaptation_weights:
                    current_weight = self.adaptation_weights[component]
                    # Increase weight if performance is good, decrease if bad
                    adjustment = (feedback_score - 0.5) * 0.05  # Small adjustment
                    new_weight = max(0.05, min(0.5, current_weight + adjustment))
                    self.adaptation_weights[component] = new_weight
            
            # Normalize weights
            total_weight = sum(self.adaptation_weights.values())
            for component in self.adaptation_weights:
                self.adaptation_weights[component] /= total_weight
            
            logger.info(f"Updated adaptation weights: {self.adaptation_weights}")
    
    def get_adaptation_statistics(self) -> Dict:
        """Get statistics about the adaptive selection process"""
        with self.lock:
            stats = {
                'total_instances': len(self.instances),
                'total_patterns': len(self.context_patterns),
                'selection_history_size': len(self.selection_history),
                'adaptation_weights': self.adaptation_weights.copy(),
                'instance_stats': {}
            }
            
            for name, instance in self.instances.items():
                stats['instance_stats'][name] = {
                    'selection_count': instance.selection_count,
                    'last_selected': instance.last_selected,
                    'adaptation_score': instance.adaptation_score,
                    'performance_history_size': len(instance.performance_history),
                    'context_affinities': len(instance.context_affinity)
                }
            
            return stats

class AdaptiveOperator:
    """Kubernetes operator for adaptive context-based instance selection"""
    
    def __init__(self):
        self.selector = AdaptiveContextSelector()
        self.k8s_client = None
        self.setup_kubernetes()
    
    def setup_kubernetes(self):
        """Initialize Kubernetes client"""
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.k8s_client = client.ApiClient()
    
    async def discover_and_register_instances(self, service_name: str, namespace: str):
        """Discover and register service instances"""
        v1 = client.CoreV1Api(self.k8s_client)
        apps_v1 = client.AppsV1Api(self.k8s_client)
        
        try:
            # Get pods for the service
            pods = v1.list_namespaced_pod(
                namespace,
                label_selector=f"app={service_name}"
            )
            
            for pod in pods.items:
                if pod.status.phase == "Running":
                    instance_context = InstanceContext(
                        name=pod.metadata.name,
                        namespace=namespace,
                        endpoint=f"{pod.status.pod_ip}:8080",
                        zone=pod.metadata.labels.get('topology.kubernetes.io/zone', 'unknown'),
                        capabilities=self._extract_capabilities(pod),
                        resource_profile=self._extract_resource_profile(pod)
                    )
                    
                    self.selector.register_instance(instance_context)
                    
                    # Simulate performance metrics collection
                    await self._collect_performance_metrics(instance_context)
                    
        except ApiException as e:
            logger.error(f"Error discovering instances: {e}")
    
    def _extract_capabilities(self, pod) -> List[str]:
        """Extract capabilities from pod annotations"""
        capabilities = []
        annotations = pod.metadata.annotations or {}
        
        if 'capabilities' in annotations:
            capabilities = annotations['capabilities'].split(',')
        
        # Add default capabilities based on labels
        labels = pod.metadata.labels or {}
        if 'tier' in labels:
            capabilities.append(f"tier-{labels['tier']}")
        if 'version' in labels:
            capabilities.append(f"version-{labels['version']}")
        
        return capabilities
    
    def _extract_resource_profile(self, pod) -> Dict[str, float]:
        """Extract resource profile from pod spec"""
        profile = {
            'cpu_capacity': 1.0,
            'memory_capacity': 1.0,
            'network_bandwidth': 1.0
        }
        
        if pod.spec.containers:
            container = pod.spec.containers[0]
            if container.resources and container.resources.limits:
                limits = container.resources.limits
                if 'cpu' in limits:
                    profile['cpu_capacity'] = float(limits['cpu'].rstrip('m')) / 1000
                if 'memory' in limits:
                    profile['memory_capacity'] = float(limits['memory'].rstrip('Mi')) / 1024
        
        return profile
    
    async def _collect_performance_metrics(self, instance: InstanceContext):
        """Collect performance metrics for an instance"""
        # Simulate metrics collection with contextual factors
        import random
        
        current_time = time.time()
        hour = int((current_time % 86400) // 3600)
        day_of_week = int((current_time // 86400) % 7)
        
        # Create contextual factors
        context = ContextualFactors(
            timestamp=current_time,
            request_type=random.choice(['api', 'web', 'batch', 'streaming']),
            user_location=random.choice(['us-east', 'us-west', 'eu-central', 'asia-pacific']),
            time_of_day=hour,
            day_of_week=day_of_week,
            traffic_load=random.uniform(0.1, 1.0),
            network_conditions=random.choice(['excellent', 'good', 'fair', 'poor']),
            user_priority=random.choice(['high', 'medium', 'low']),
            session_type=random.choice(['interactive', 'batch', 'realtime']),
            data_sensitivity=random.choice(['public', 'internal', 'confidential']),
            geographical_zone=instance.zone,
            device_type=random.choice(['mobile', 'desktop', 'tablet', 'iot'])
        )
        
        # Simulate performance metrics influenced by context
        base_performance = random.uniform(0.5, 0.9)
        time_factor = 1.0 - 0.3 * abs(hour - 12) / 12  # Better during business hours
        traffic_factor = 1.0 - 0.4 * context.traffic_load  # Worse with high traffic
        
        performance_factor = base_performance * time_factor * traffic_factor
        
        metrics = PerformanceMetrics(
            response_time=max(10, 200 * (1 - performance_factor) + random.gauss(0, 20)),
            throughput=max(1, 300 * performance_factor + random.gauss(0, 30)),
            error_rate=max(0, 5 * (1 - performance_factor) + random.gauss(0, 1)),
            cpu_usage=max(0, min(100, 50 + 30 * (1 - performance_factor) + random.gauss(0, 10))),
            memory_usage=max(0, min(100, 40 + 35 * (1 - performance_factor) + random.gauss(0, 8))),
            network_latency=max(5, 100 * (1 - performance_factor) + random.gauss(0, 15)),
            success_rate=min(100, 95 + 5 * performance_factor + random.gauss(0, 2)),
            context=context
        )
        
        self.selector.update_performance_metrics(instance.name, metrics)
    
    def _create_context_from_request(self, request_spec: Dict) -> ContextualFactors:
        """Create context from request specification"""
        current_time = time.time()
        hour = int((current_time % 86400) // 3600)
        day_of_week = int((current_time // 86400) % 7)
        
        return ContextualFactors(
            timestamp=current_time,
            request_type=request_spec.get('requestType', 'api'),
            user_location=request_spec.get('userLocation', 'unknown'),
            time_of_day=hour,
            day_of_week=day_of_week,
            traffic_load=request_spec.get('trafficLoad', 0.5),
            network_conditions=request_spec.get('networkConditions', 'good'),
            user_priority=request_spec.get('userPriority', 'medium'),
            session_type=request_spec.get('sessionType', 'interactive'),
            data_sensitivity=request_spec.get('dataSensitivity', 'internal'),
            geographical_zone=request_spec.get('geographicalZone', 'unknown'),
            device_type=request_spec.get('deviceType', 'desktop')
        )

# Kopf handlers for the operator
@kopf.on.startup()
async def configure(settings: kopf.OperatorSettings, **_):
    """Configure the operator"""
    settings.posting.level = logging.INFO
    logger.info("Adaptive Context-Based Instance Selection Operator starting up")

@kopf.on.create('microservice-selector.io', 'v1', 'adaptiveselectors')
async def create_adaptive_selector(spec, name, namespace, **kwargs):
    """Handle creation of AdaptiveSelector custom resource"""
    logger.info(f"Creating adaptive selector: {name}")
    
    operator = AdaptiveOperator()
    service_name = spec.get('serviceName')
    
    # Discover and register instances
    await operator.discover_and_register_instances(service_name, namespace)
    
    # Create context from request
    request_context = spec.get('requestContext', {})
    context = operator._create_context_from_request(request_context)
    
    # Select optimal instance
    selected_instance = operator.selector.select_optimal_instance(context)
    
    if selected_instance:
        stats = operator.selector.get_adaptation_statistics()
        
        return {
            'selectedInstance': {
                'name': selected_instance.name,
                'endpoint': selected_instance.endpoint,
                'zone': selected_instance.zone,
                'adaptationScore': selected_instance.adaptation_score,
                'selectionCount': selected_instance.selection_count,
                'capabilities': selected_instance.capabilities
            },
            'contextAnalysis': {
                'requestType': context.request_type,
                'trafficLoad': context.traffic_load,
                'timeOfDay': context.time_of_day,
                'userLocation': context.user_location,
                'geographicalZone': context.geographical_zone
            },
            'adaptationStats': stats,
            'algorithm': 'adaptive-context-based',
            'selectionTime': time.time()
        }
    
    return {'status': 'no instances available'}

@kopf.on.update('microservice-selector.io', 'v1', 'adaptiveselectors')
async def update_adaptive_selector(spec, name, namespace, **kwargs):
    """Handle updates to AdaptiveSelector custom resource"""
    logger.info(f"Updating adaptive selector: {name}")
    return await create_adaptive_selector(spec, name, namespace, **kwargs)

@kopf.timer('microservice-selector.io', 'v1', 'adaptiveselectors', interval=25.0)
async def periodic_adaptive_selection(spec, name, namespace, **kwargs):
    """Periodically re-evaluate instance selection with adaptation"""
    logger.info(f"Periodic adaptive selection for: {name}")
    return await create_adaptive_selector(spec, name, namespace, **kwargs)

@kopf.on.field('microservice-selector.io', 'v1', 'adaptiveselectors', field='spec.feedback')
async def handle_feedback(spec, name, namespace, old, new, **kwargs):
    """Handle performance feedback for adaptation"""
    if new and new != old:
        logger.info(f"Received feedback for adaptive selector: {name}")
        
        operator = AdaptiveOperator()
        feedback = new.get('performanceFeedback', {})
        
        if feedback:
            operator.selector.adapt_selection_weights(feedback)
            logger.info("Adapted selection weights based on feedback")

if __name__ == '__main__':
    # Run the operator
    kopf.run()