# @Author: Swastik N. (2025)

import os
import json
import random
import asyncio
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import psutil
import sys
import time

from .resource_monitor import ResourceMonitor
from .ipc_tester import IPCTester
from .protocol_parser import ProtocolParser
from .notification_system import NotificationSystem
from .load_test_scenario import LoadTestScenario
from .microservice_algorithms import (
    LoadBalancer, ServiceInstance, RequestContext, QoSMetrics, ServiceMeshType
)

import logging
logger = logging.getLogger(__name__)

class ComprehensiveTestSuite:
    """Main test suite orchestrator"""
    
    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.ipc_tester = IPCTester()
        self.protocol_parser = ProtocolParser()
        self.notification_system = NotificationSystem()
        self.test_results = defaultdict(list)
        self.mesh_types = [ServiceMeshType.ISTIO, ServiceMeshType.ISTIO_AMBIENT, ServiceMeshType.LINKERD]
        
    def setup_test_environment(self, scenario: Dict[str, Any], mesh_type: ServiceMeshType):
        """Setup test environment with services and instances"""
        load_balancer = LoadBalancer(mesh_type)
        
        # Create service instances
        for service_idx in range(scenario['num_services']):
            service_name = f"service-{service_idx}"
            instances = []
            
            for instance_idx in range(scenario['instances_per_service']):
                instance = ServiceInstance(
                    id=f"{service_name}-{instance_idx}",
                    service_name=service_name,
                    endpoint=f"http://10.0.{service_idx}.{instance_idx}:8080",
                    zone=random.choice(['us-central1-a', 'us-central1-b', 'us-central1-c']),
                    node_name=f"node-{instance_idx % 3}",
                    pod_name=f"{service_name}-pod-{instance_idx}",
                    namespace="default",
                    metrics=QoSMetrics(
                        response_time=random.uniform(0.01, 0.2),
                        throughput=random.uniform(100, 1000),
                        availability=random.uniform(0.95, 1.0),
                        cpu_usage=random.uniform(0.1, 0.8),
                        memory_usage=random.uniform(0.2, 0.7),
                        network_latency=random.uniform(0.001, 0.01),
                        error_rate=random.uniform(0.0, 0.05)
                    ),
                    tags={'optimized_for': random.choice(['read', 'write', 'compute'])}
                )
                instances.append(instance)
            
            load_balancer.register_service_instances(service_name, instances)
        
        return load_balancer
    
    def generate_request_contexts(self, scenario: Dict[str, Any]) -> List[RequestContext]:
        """Generate realistic request contexts"""
        contexts = []
        
        for i in range(scenario['user_sessions']):
            for j in range(random.randint(1, 10)):  # Requests per session
                context = RequestContext(
                    user_id=f"user-{i}",
                    session_id=f"session-{i}",
                    request_type=random.choice(['read', 'write', 'compute', 'stream']),
                    priority=random.randint(1, 5),
                    timeout=random.uniform(1.0, 30.0),
                    source_service=f"service-{random.randint(0, scenario['num_services']-1)}",
                    target_service=f"service-{random.randint(0, scenario['num_services']-1)}",
                    geographic_region=random.choice(['us-central1', 'us-east1', 'europe-west1']),
                    headers={'Content-Type': 'application/json', 'X-Request-ID': f'req-{i}-{j}'}
                )
                contexts.append(context)
        
        return contexts
    
    async def run_algorithm_benchmark(self, algorithm_name: str, load_balancer: LoadBalancer,
                                    contexts: List[RequestContext], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive benchmark for a specific algorithm"""
        logger.info(f"Starting benchmark for {algorithm_name} algorithm")
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Performance tracking
        start_time = time.time()
        successful_requests = 0
        failed_requests = 0
        response_times = []
        selection_times = []
        
        # Concurrent request processing
        semaphore = asyncio.Semaphore(scenario['concurrent_requests'])
        
        async def process_request(context: RequestContext):
            async with semaphore:
                try:
                    # Measure selection time
                    selection_start = time.perf_counter()
                    
                    # Select instance
                    service_name = context.target_service
                    selected_instance = load_balancer.select_instance(
                        service_name, context, algorithm_name
                    )
                    
                    selection_time = (time.perf_counter() - selection_start) * 1000
                    selection_times.append(selection_time)
                    
                    if selected_instance:
                        # Simulate request processing
                        processing_time = random.uniform(0.01, 0.5)
                        await asyncio.sleep(processing_time)
                        
                        # Simulate response
                        response_time = processing_time + selected_instance.metrics.response_time
                        response_times.append(response_time * 1000)  # Convert to ms
                        
                        # Update metrics
                        success = random.random() > selected_instance.metrics.error_rate
                        load_balancer.update_instance_metrics(
                            selected_instance.id, service_name, response_time, success, algorithm_name
                        )
                        
                        if success:
                            nonlocal successful_requests
                            successful_requests += 1
                        else:
                            nonlocal failed_requests
                            failed_requests += 1
                    else:
                        failed_requests += 1
                        
                except Exception as e:
                    logger.error(f"Request processing error: {e}")
                    failed_requests += 1
        
        # Rate limiting for realistic load
        request_interval = 1.0 / scenario['request_rate_per_second']
        
        tasks = []
        for i, context in enumerate(contexts):
            if i > 0:
                await asyncio.sleep(request_interval)
            
            task = asyncio.create_task(process_request(context))
            tasks.append(task)
            
            # Check if test duration exceeded
            if time.time() - start_time > scenario['test_duration_seconds']:
                break
        
        # Wait for all requests to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Capture final metrics
        final_metrics = self.resource_monitor.capture_metrics(algorithm_name, scenario['name'])
        self.resource_monitor.stop_monitoring()
        
        # Calculate performance statistics
        total_requests = successful_requests + failed_requests
        success_rate = successful_requests / max(1, total_requests)
        throughput = total_requests / total_time
        avg_response_time = statistics.mean(response_times) if response_times else 0
        avg_selection_time = statistics.mean(selection_times) if selection_times else 0
        
        return {
            'algorithm': algorithm_name,
            'scenario': scenario['name'],
            'mesh_type': load_balancer.mesh_integration.mesh_type.value,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': success_rate,
            'throughput_rps': throughput,
            'avg_response_time_ms': avg_response_time,
            'avg_selection_time_ms': avg_selection_time,
            'p95_response_time_ms': np.percentile(response_times, 95) if response_times else 0,
            'p99_response_time_ms': np.percentile(response_times, 99) if response_times else 0,
            'total_test_time_seconds': total_time,
            'cpu_usage_percent': final_metrics.cpu_percent,
            'memory_usage_mb': final_metrics.memory_usage_mb,
            'memory_peak_mb': final_metrics.memory_peak_mb,
            'disk_read_bytes': final_metrics.disk_read_bytes,
            'disk_write_bytes': final_metrics.disk_write_bytes,
            'active_threads': final_metrics.active_threads,
            'context_switches': final_metrics.context_switches
        }
    
    def run_ipc_benchmark(self) -> Dict[str, Any]:
        """Run IPC performance benchmarks"""
        logger.info("Running IPC benchmarks")
        
        results = {}
        
        # Message Queue benchmark
        results['message_queue'] = self.ipc_tester.test_message_queue_performance(1000)
        
        # Pipe benchmark
        results['pipe'] = self.ipc_tester.test_pipe_performance(1000)
        
        # Shared Memory benchmark
        results['shared_memory'] = self.ipc_tester.test_shared_memory_performance(10000)
        
        return results
    
    def run_protocol_parsing_benchmark(self) -> Dict[str, Any]:
        """Run protocol parsing benchmarks"""
        logger.info("Running protocol parsing benchmarks")
        
        results = {}
        
        # HTTP/1.1 parsing
        results['http1'] = self.protocol_parser.benchmark_parsing('HTTP/1.1', 1000)
        
        # gRPC parsing
        results['grpc'] = self.protocol_parser.benchmark_parsing('gRPC', 1000)
        
        return results
    
    def run_notification_benchmark(self) -> Dict[str, Any]:
        """Run notification system benchmarks"""
        logger.info("Running notification system benchmarks")
        
        return self.notification_system.benchmark_notifications(1000, 50)
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        logger.info("Starting comprehensive test suite")
        
        all_results = {
            'test_timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version,
                'platform': os.name
            },
            'scenarios': {},
            'ipc_results': {},
            'protocol_parsing_results': {},
            'notification_results': {}
        }
        
        # Test scenarios
        scenarios = [
            LoadTestScenario.light_load_scenario(),
            LoadTestScenario.medium_load_scenario(),
            LoadTestScenario.heavy_load_scenario(),
            LoadTestScenario.burst_load_scenario()
        ]
        
        algorithms = ['fuzzy', 'stochastic', 'adaptive']
        
        # Run algorithm benchmarks for each scenario and mesh type
        for scenario in scenarios:
            scenario_results = {}
            logger.info(f"Testing scenario: {scenario['name']}")
            
            for mesh_type in self.mesh_types:
                mesh_results = {}
                logger.info(f"Testing with mesh: {mesh_type.value}")
                
                # Setup environment
                load_balancer = self.setup_test_environment(scenario, mesh_type)
                contexts = self.generate_request_contexts(scenario)
                
                for algorithm in algorithms:
                    try:
                        result = await self.run_algorithm_benchmark(
                            algorithm, load_balancer, contexts, scenario
                        )
                        mesh_results[algorithm] = result
                        
                        # Small delay between algorithms to prevent interference
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"Error testing {algorithm} with {mesh_type.value}: {e}")
                        mesh_results[algorithm] = {'error': str(e)}
                
                scenario_results[mesh_type.value] = mesh_results
            
            all_results['scenarios'][scenario['name']] = scenario_results
        
        # Run auxiliary benchmarks
        try:
            all_results['ipc_results'] = self.run_ipc_benchmark()
        except Exception as e:
            logger.error(f"IPC benchmark failed: {e}")
            all_results['ipc_results'] = {'error': str(e)}
        
        try:
            all_results['protocol_parsing_results'] = self.run_protocol_parsing_benchmark()
        except Exception as e:
            logger.error(f"Protocol parsing benchmark failed: {e}")
            all_results['protocol_parsing_results'] = {'error': str(e)}
        
        try:
            all_results['notification_results'] = self.run_notification_benchmark()
        except Exception as e:
            logger.error(f"Notification benchmark failed: {e}")
            all_results['notification_results'] = {'error': str(e)}
        
        return all_results
    
    def generate_performance_report(self, results: Dict[str, Any], output_dir: str = './test_results'):
        """Generate comprehensive performance report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        with open(f"{output_dir}/raw_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate CSV reports
        self._generate_csv_reports(results, output_dir)
        
        # Generate performance comparison charts
        self._generate_performance_charts(results, output_dir)
        
        # Generate summary report
        self._generate_summary_report(results, output_dir)
        
        logger.info(f"Performance report generated in {output_dir}")
    
    def _generate_csv_reports(self, results: Dict[str, Any], output_dir: str):
        """Generate CSV reports for detailed analysis"""
        
        # Algorithm performance comparison
        algorithm_data = []
        for scenario_name, scenario_results in results['scenarios'].items():
            for mesh_type, mesh_results in scenario_results.items():
                for algorithm, alg_results in mesh_results.items():
                    if 'error' not in alg_results:
                        row = {
                            'scenario': scenario_name,
                            'mesh_type': mesh_type,
                            'algorithm': algorithm,
                            **alg_results
                        }
                        algorithm_data.append(row)
        
        if algorithm_data:
            df = pd.DataFrame(algorithm_data)
            df.to_csv(f"{output_dir}/algorithm_performance.csv", index=False)
        
        # IPC performance
        if 'ipc_results' in results and 'error' not in results['ipc_results']:
            ipc_data = []
            for ipc_type, ipc_metrics in results['ipc_results'].items():
                row = {'ipc_type': ipc_type, **ipc_metrics}
                ipc_data.append(row)
            
            df_ipc = pd.DataFrame(ipc_data)
            df_ipc.to_csv(f"{output_dir}/ipc_performance.csv", index=False)
        
        # Protocol parsing performance
        if 'protocol_parsing_results' in results and 'error' not in results['protocol_parsing_results']:
            protocol_data = []
            for protocol, metrics in results['protocol_parsing_results'].items():
                row = {'protocol': protocol, **metrics}
                protocol_data.append(row)
            
            df_protocol = pd.DataFrame(protocol_data)
            df_protocol.to_csv(f"{output_dir}/protocol_parsing_performance.csv", index=False)
    
    def _generate_performance_charts(self, results: Dict[str, Any], output_dir: str):
        """Generate performance visualization charts"""
        plt.style.use('seaborn-v0_8')
        
        # Algorithm throughput comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Microservice Instance Selection Algorithm Performance Comparison', fontsize=16)
        
        # Collect data for plotting
        scenarios = []
        algorithms = []
        throughputs = []
        response_times = []
        cpu_usages = []
        memory_usages = []
        mesh_types = []
        
        for scenario_name, scenario_results in results['scenarios'].items():
            for mesh_type, mesh_results in scenario_results.items():
                for algorithm, alg_results in mesh_results.items():
                    if 'error' not in alg_results:
                        scenarios.append(scenario_name)
                        algorithms.append(algorithm)
                        throughputs.append(alg_results.get('throughput_rps', 0))
                        response_times.append(alg_results.get('avg_response_time_ms', 0))
                        cpu_usages.append(alg_results.get('cpu_usage_percent', 0))
                        memory_usages.append(alg_results.get('memory_usage_mb', 0))
                        mesh_types.append(mesh_type)
        
        if scenarios:
            df_plot = pd.DataFrame({
                'scenario': scenarios,
                'algorithm': algorithms,
                'throughput_rps': throughputs,
                'response_time_ms': response_times,
                'cpu_usage_percent': cpu_usages,
                'memory_usage_mb': memory_usages,
                'mesh_type': mesh_types
            })
            
            # Throughput comparison
            sns.barplot(data=df_plot, x='scenario', y='throughput_rps', hue='algorithm', ax=axes[0,0])
            axes[0,0].set_title('Throughput Comparison (Requests/Second)')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Response time comparison
            sns.barplot(data=df_plot, x='scenario', y='response_time_ms', hue='algorithm', ax=axes[0,1])
            axes[0,1].set_title('Average Response Time (ms)')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # CPU usage comparison
            sns.barplot(data=df_plot, x='scenario', y='cpu_usage_percent', hue='algorithm', ax=axes[1,0])
            axes[1,0].set_title('CPU Usage (%)')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Memory usage comparison
            sns.barplot(data=df_plot, x='scenario', y='memory_usage_mb', hue='algorithm', ax=axes[1,1])
            axes[1,1].set_title('Memory Usage (MB)')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/algorithm_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Service mesh comparison
        if mesh_types:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Service Mesh Performance Impact', fontsize=16)
            
            # Throughput by mesh type
            mesh_throughput = df_plot.groupby(['mesh_type'])['throughput_rps'].mean()
            mesh_throughput.plot(kind='bar', ax=axes[0])
            axes[0].set_title('Average Throughput by Mesh Type')
            axes[0].set_ylabel('Requests/Second')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Response time by mesh type
            mesh_response_time = df_plot.groupby(['mesh_type'])['response_time_ms'].mean()
            mesh_response_time.plot(kind='bar', ax=axes[1])
            axes[1].set_title('Average Response Time by Mesh Type')
            axes[1].set_ylabel('Response Time (ms)')
            axes[1].tick_params(axis='x', rotation=45)
            
            # CPU usage by mesh type
            mesh_cpu = df_plot.groupby(['mesh_type'])['cpu_usage_percent'].mean()
            mesh_cpu.plot(kind='bar', ax=axes[2])
            axes[2].set_title('Average CPU Usage by Mesh Type')
            axes[2].set_ylabel('CPU Usage (%)')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/mesh_performance_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # IPC and Protocol parsing charts
        if 'ipc_results' in results and 'error' not in results['ipc_results']:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # IPC latency comparison
            ipc_types = list(results['ipc_results'].keys())
            ipc_latencies = [results['ipc_results'][ipc_type].get('latency_per_message_ms', 0) 
                           for ipc_type in ipc_types]
            
            axes[0].bar(ipc_types, ipc_latencies)
            axes[0].set_title('IPC Latency Comparison')
            axes[0].set_ylabel('Latency per Message (ms)')
            axes[0].tick_params(axis='x', rotation=45)
            
            # IPC throughput comparison
            ipc_throughputs = [results['ipc_results'][ipc_type].get('messages_per_second', 0) 
                             for ipc_type in ipc_types]
            
            axes[1].bar(ipc_types, ipc_throughputs)
            axes[1].set_title('IPC Throughput Comparison')
            axes[1].set_ylabel('Messages per Second')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/ipc_performance.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_summary_report(self, results: Dict[str, Any], output_dir: str):
        """Generate a comprehensive summary report"""
        report_lines = []
        report_lines.append("# Microservice Instance Selection Algorithm Performance Report")
        report_lines.append(f"Generated on: {results['test_timestamp']}")
        report_lines.append("")
        
        # System information
        report_lines.append("## System Information")
        sys_info = results['system_info']
        report_lines.append(f"- CPU Cores: {sys_info['cpu_count']}")
        report_lines.append(f"- Total Memory: {sys_info['memory_total_gb']:.2f} GB")
        report_lines.append(f"- Python Version: {sys_info['python_version']}")
        report_lines.append(f"- Platform: {sys_info['platform']}")
        report_lines.append("")
        
        # Algorithm performance summary
        report_lines.append("## Algorithm Performance Summary")
        
        # Calculate overall statistics
        all_results = []
        for scenario_name, scenario_results in results['scenarios'].items():
            for mesh_type, mesh_results in scenario_results.items():
                for algorithm, alg_results in mesh_results.items():
                    if 'error' not in alg_results:
                        all_results.append({
                            'scenario': scenario_name,
                            'mesh': mesh_type,
                            'algorithm': algorithm,
                            **alg_results
                        })
        
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Best performing algorithm by metric
            best_throughput = df.loc[df['throughput_rps'].idxmax()]
            best_response_time = df.loc[df['avg_response_time_ms'].idxmin()]
            best_cpu_efficiency = df.loc[df['cpu_usage_percent'].idxmin()]
            best_memory_efficiency = df.loc[df['memory_usage_mb'].idxmin()]
            
            report_lines.append("### Best Performing Algorithms")
            report_lines.append(f"- **Highest Throughput**: {best_throughput['algorithm']} "
                              f"({best_throughput['throughput_rps']:.2f} RPS in "
                              f"{best_throughput['scenario']} with {best_throughput['mesh']})")
            
            report_lines.append(f"- **Lowest Response Time**: {best_response_time['algorithm']} "
                              f"({best_response_time['avg_response_time_ms']:.2f} ms in "
                              f"{best_response_time['scenario']} with {best_response_time['mesh']})")
            
            report_lines.append(f"- **Lowest CPU Usage**: {best_cpu_efficiency['algorithm']} "
                              f"({best_cpu_efficiency['cpu_usage_percent']:.2f}% in "
                              f"{best_cpu_efficiency['scenario']} with {best_cpu_efficiency['mesh']})")
            
            report_lines.append(f"- **Lowest Memory Usage**: {best_memory_efficiency['algorithm']} "
                              f"({best_memory_efficiency['memory_usage_mb']:.2f} MB in "
                              f"{best_memory_efficiency['scenario']} with {best_memory_efficiency['mesh']})")
            report_lines.append("")
            
            # Algorithm comparison table
            report_lines.append("### Algorithm Comparison (Average Performance)")
            algo_summary = df.groupby('algorithm').agg({
                'throughput_rps': 'mean',
                'avg_response_time_ms': 'mean',
                'success_rate': 'mean',
                'cpu_usage_percent': 'mean',
                'memory_usage_mb': 'mean',
                'avg_selection_time_ms': 'mean'
            }).round(2)
            
            report_lines.append("| Algorithm | Throughput (RPS) | Response Time (ms) | Success Rate | CPU Usage (%) | Memory (MB) | Selection Time (ms) |")
            report_lines.append("|-----------|------------------|-------------------|--------------|---------------|-------------|-------------------|")
            
            for algorithm, row in algo_summary.iterrows():
                report_lines.append(f"| {algorithm} | {row['throughput_rps']} | "
                                  f"{row['avg_response_time_ms']} | {row['success_rate']:.3f} | "
                                  f"{row['cpu_usage_percent']} | {row['memory_usage_mb']} | "
                                  f"{row['avg_selection_time_ms']} |")
            report_lines.append("")
            
            # Service mesh comparison
            report_lines.append("### Service Mesh Performance Impact")
            mesh_summary = df.groupby('mesh').agg({
                'throughput_rps': 'mean',
                'avg_response_time_ms': 'mean',
                'cpu_usage_percent': 'mean',
                'memory_usage_mb': 'mean'
            }).round(2)
            
            report_lines.append("| Mesh Type | Throughput (RPS) | Response Time (ms) | CPU Usage (%) | Memory (MB) |")
            report_lines.append("|-----------|------------------|-------------------|---------------|-------------|")
            
            for mesh, row in mesh_summary.iterrows():
                report_lines.append(f"| {mesh} | {row['throughput_rps']} | "
                                  f"{row['avg_response_time_ms']} | {row['cpu_usage_percent']} | "
                                  f"{row['memory_usage_mb']} |")
            report_lines.append("")
        
        # IPC Performance
        if 'ipc_results' in results and 'error' not in results['ipc_results']:
            report_lines.append("## IPC Performance Results")
            for ipc_type, metrics in results['ipc_results'].items():
                report_lines.append(f"### {ipc_type.replace('_', ' ').title()}")
                report_lines.append(f"- Messages per Second: {metrics.get('messages_per_second', 0):.2f}")
                report_lines.append(f"- Latency per Message: {metrics.get('latency_per_message_ms', 0):.4f} ms")
                report_lines.append(f"- Total Time: {metrics.get('total_time_ms', 0):.2f} ms")
                report_lines.append("")
        
        # Protocol Parsing Performance
        if 'protocol_parsing_results' in results and 'error' not in results['protocol_parsing_results']:
            report_lines.append("## Protocol Parsing Performance")
            for protocol, metrics in results['protocol_parsing_results'].items():
                report_lines.append(f"### {protocol}")
                report_lines.append(f"- Messages per Second: {metrics.get('messages_per_second', 0):.2f}")
                report_lines.append(f"- Average Parse Time: {metrics.get('avg_parse_time_ms', 0):.4f} ms")
                report_lines.append(f"- Parse Errors: {metrics.get('parse_errors', 0)}")
                report_lines.append("")
        
        # Notification System Performance
        if 'notification_results' in results and 'error' not in results['notification_results']:
            report_lines.append("## Notification System Performance")
            notif_metrics = results['notification_results']
            report_lines.append(f"- Notifications per Second: {notif_metrics.get('notifications_per_second', 0):.2f}")
            report_lines.append(f"- Average Notification Time: {notif_metrics.get('avg_notification_time_ms', 0):.4f} ms")
            report_lines.append(f"- Subscribers: {notif_metrics.get('subscribers', 0)}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        if all_results:
            # Analyze results and provide recommendations
            avg_performance = df.groupby('algorithm').agg({
                'throughput_rps': 'mean',
                'avg_response_time_ms': 'mean',
                'cpu_usage_percent': 'mean',
                'success_rate': 'mean'
            })
            
            best_overall = avg_performance.loc[
                (avg_performance['success_rate'] > 0.95) & 
                (avg_performance['cpu_usage_percent'] < avg_performance['cpu_usage_percent'].median())
            ].sort_values('throughput_rps', ascending=False)
            
            if not best_overall.empty:
                best_algorithm = best_overall.index[0]
                report_lines.append(f"- **For production use**: {best_algorithm} algorithm shows the best "
                                  f"balance of performance and resource efficiency")
            
            # Mesh recommendations
            mesh_perf = df.groupby('mesh')['throughput_rps'].mean().sort_values(ascending=False)
            if not mesh_perf.empty:
                best_mesh = mesh_perf.index[0]
                report_lines.append(f"- **Service Mesh**: {best_mesh} provides the best overall performance")
            
            report_lines.append("- Monitor CPU and memory usage in production environments")
            report_lines.append("- Consider load balancing algorithms based on specific use case requirements")
            report_lines.append("- Implement proper monitoring and alerting for instance selection performance")
        
        # Write the report
        with open(f"{output_dir}/performance_summary.md", 'w') as f:
            f.write('\n'.join(report_lines))