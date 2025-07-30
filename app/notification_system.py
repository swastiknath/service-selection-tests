import time
import random
import queue
import statistics
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class NotificationSystem:
    """Simulate notification system performance"""
    
    def __init__(self):
        self.notification_queue = queue.Queue()
        self.subscribers = {}
        self.notification_times = []
        
    def subscribe(self, subscriber_id: str, callback):
        """Subscribe to notifications"""
        self.subscribers[subscriber_id] = callback
        
    def publish_notification(self, notification: Dict[str, Any]) -> float:
        """Publish notification and measure time"""
        start_time = time.perf_counter()
        
        # Add to queue
        self.notification_queue.put(notification)
        
        # Notify all subscribers
        for subscriber_id, callback in self.subscribers.items():
            try:
                callback(notification)
            except Exception as e:
                logger.error(f"Notification callback error for {subscriber_id}: {e}")
        
        processing_time = (time.perf_counter() - start_time) * 1000
        self.notification_times.append(processing_time)
        
        return processing_time
    
    def benchmark_notifications(self, num_notifications: int = 1000, 
                              num_subscribers: int = 10) -> Dict[str, float]:
        """Benchmark notification system performance"""
        
        # Setup subscribers
        def dummy_callback(notification):
            # Simulate processing time
            time.sleep(0.0001)  # 0.1ms processing time
        
        for i in range(num_subscribers):
            self.subscribe(f"subscriber_{i}", dummy_callback)
        
        start_time = time.time()
        
        # Send notifications
        for i in range(num_notifications):
            notification = {
                'id': i,
                'type': 'instance_selection',
                'timestamp': time.time(),
                'data': {
                    'instance_id': f'instance_{i % 5}',
                    'algorithm': 'adaptive',
                    'response_time': random.uniform(0.01, 0.1)
                }
            }
            self.publish_notification(notification)
        
        total_time = time.time() - start_time
        
        return {
            'total_time_ms': total_time * 1000,
            'notifications_sent': num_notifications,
            'subscribers': num_subscribers,
            'notifications_per_second': num_notifications / total_time,
            'avg_notification_time_ms': statistics.mean(self.notification_times),
            'max_notification_time_ms': max(self.notification_times),
            'min_notification_time_ms': min(self.notification_times)
        }