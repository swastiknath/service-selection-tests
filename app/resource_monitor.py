import psutil
import tracemalloc
import time
from collections import deque
# @Author: Swastik N. (2025)
import os
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    algorithm_name: str
    test_name: str
    timestamp: float
    cpu_percent: float
    memory_usage_mb: float
    memory_peak_mb: float
    cpu_user_time: float
    cpu_system_time: float
    disk_read_bytes: int
    disk_write_bytes: int
    active_threads: int
    context_switches: int

class ResourceMonitor:
    """Monitor system resources during testing"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics_history = deque(maxlen=10000)
        self.process = psutil.Process()
        self.initial_io_counters = None
        self.initial_cpu_times = None
        
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.initial_io_counters = self.process.io_counters()
        self.initial_cpu_times = self.process.cpu_times()
        tracemalloc.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        tracemalloc.stop()
        
    def capture_metrics(self, algorithm_name: str, test_name: str) -> PerformanceMetrics:
        """Capture current resource metrics"""
        current_time = time.time()
        
        # CPU and Memory
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        
        cpu_times = self.process.cpu_times()
        cpu_user_time = cpu_times.user - (self.initial_cpu_times.user if self.initial_cpu_times else 0)
        cpu_system_time = cpu_times.system - (self.initial_cpu_times.system if self.initial_cpu_times else 0)
        
        # I/O counters
        io_counters = self.process.io_counters()
        disk_read_bytes = io_counters.read_bytes - (self.initial_io_counters.read_bytes if self.initial_io_counters else 0)
        disk_write_bytes = io_counters.write_bytes - (self.initial_io_counters.write_bytes if self.initial_io_counters else 0)
        
        # Memory peak
        if tracemalloc.is_tracing():
            _, peak_memory = tracemalloc.get_traced_memory()
            memory_peak_mb = peak_memory / 1024 / 1024
        else:
            memory_peak_mb = memory_usage_mb
        
        # Thread count
        active_threads = self.process.num_threads()
        
        # Context switches
        ctx_switches = self.process.num_ctx_switches()
        context_switches = ctx_switches.voluntary + ctx_switches.involuntary
        
        metrics = PerformanceMetrics(
            algorithm_name=algorithm_name,
            test_name=test_name,
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_usage_mb=memory_usage_mb,
            memory_peak_mb=memory_peak_mb,
            cpu_user_time=cpu_user_time,
            cpu_system_time=cpu_system_time,
            disk_read_bytes=disk_read_bytes,
            disk_write_bytes=disk_write_bytes,
            active_threads=active_threads,
            context_switches=context_switches
        )
        
        self.metrics_history.append(metrics)
        return metrics