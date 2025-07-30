import time
import random
import multiprocessing
import concurrent.futures
from typing import Dict
# @Author: Swastik N. (2025)


class IPCTester:
    """Test Inter-Process Communication performance"""
    
    def __init__(self):
        self.message_queue = multiprocessing.Queue()
        self.pipe_conn1, self.pipe_conn2 = multiprocessing.Pipe()
        self.shared_memory = multiprocessing.Array('i', range(1000))
        
    def test_message_queue_performance(self, num_messages: int = 1000) -> Dict[str, float]:
        """Test message queue performance"""
        start_time = time.time()
        
        # Send messages
        for i in range(num_messages):
            message = {
                'id': i,
                'timestamp': time.time(),
                'data': f'test_message_{i}' * 10  # Variable size messages
            }
            self.message_queue.put(message)
        
        send_time = time.time() - start_time
        
        # Receive messages
        start_time = time.time()
        received_count = 0
        while received_count < num_messages:
            try:
                message = self.message_queue.get(timeout=1)
                received_count += 1
            except:
                break
                
        receive_time = time.time() - start_time
        
        return {
            'send_time_ms': send_time * 1000,
            'receive_time_ms': receive_time * 1000,
            'total_time_ms': (send_time + receive_time) * 1000,
            'messages_per_second': num_messages / (send_time + receive_time),
            'latency_per_message_ms': ((send_time + receive_time) * 1000) / num_messages
        }
    
    def test_pipe_performance(self, num_messages: int = 1000) -> Dict[str, float]:
        """Test pipe communication performance"""
        def sender():
            start_time = time.time()
            for i in range(num_messages):
                message = f'pipe_message_{i}_' + 'x' * 100
                self.pipe_conn1.send(message)
            self.pipe_conn1.send('DONE')
            return time.time() - start_time
        
        def receiver():
            start_time = time.time()
            received = 0
            while True:
                message = self.pipe_conn2.recv()
                if message == 'DONE':
                    break
                received += 1
            return time.time() - start_time, received
        
        # Run sender and receiver concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            sender_future = executor.submit(sender)
            receiver_future = executor.submit(receiver)
            
            send_time = sender_future.result()
            receive_time, received_count = receiver_future.result()
        
        total_time = max(send_time, receive_time)
        
        return {
            'send_time_ms': send_time * 1000,
            'receive_time_ms': receive_time * 1000,
            'total_time_ms': total_time * 1000,
            'messages_per_second': received_count / total_time,
            'latency_per_message_ms': (total_time * 1000) / received_count
        }
    
    def test_shared_memory_performance(self, num_operations: int = 10000) -> Dict[str, float]:
        """Test shared memory performance"""
        start_time = time.time()
        
        # Write operations
        for i in range(num_operations):
            index = i % len(self.shared_memory)
            self.shared_memory[index] = random.randint(1, 1000)
        
        write_time = time.time() - start_time
        
        # Read operations
        start_time = time.time()
        total = 0
        for i in range(num_operations):
            index = i % len(self.shared_memory)
            total += self.shared_memory[index]
        
        read_time = time.time() - start_time
        
        return {
            'write_time_ms': write_time * 1000,
            'read_time_ms': read_time * 1000,
            'total_time_ms': (write_time + read_time) * 1000,
            'operations_per_second': (2 * num_operations) / (write_time + read_time),
            'checksum': total }