import time
import struct
import json
import statistics
from typing import Dict, Any, List
# @Author: Swastik N. (2025)


class ProtocolParser:
    """Simulate protocol parsing overhead"""
    
    def __init__(self):
        self.supported_protocols = ['HTTP/1.1', 'HTTP/2', 'gRPC', 'WebSocket']
        self.parse_times = []
        
    def parse_http_request(self, raw_request: str) -> Dict[str, Any]:
        """Parse HTTP request and measure time"""
        start_time = time.perf_counter()
        
        # Simulate HTTP parsing
        lines = raw_request.split('\n')
        if not lines:
            return {}
        
        # Parse request line
        request_line = lines[0].split()
        if len(request_line) < 3:
            return {}
        
        method, path, version = request_line[0], request_line[1], request_line[2]
        
        # Parse headers
        headers = {}
        body_start = 0
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '':
                body_start = i + 1
                break
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip()] = value.strip()
        
        # Parse body if present
        body = '\n'.join(lines[body_start:]) if body_start < len(lines) else ''
        
        parse_time = (time.perf_counter() - start_time) * 1000
        self.parse_times.append(parse_time)
        
        return {
            'method': method,
            'path': path,
            'version': version,
            'headers': headers,
            'body': body,
            'parse_time_ms': parse_time
        }
    
    def parse_grpc_message(self, message_data: bytes) -> Dict[str, Any]:
        """Parse gRPC message format"""
        start_time = time.perf_counter()
        
        if len(message_data) < 5:
            return {}
        
        # gRPC message format: [compression][length][data]
        compression = message_data[0]
        length = struct.unpack('>I', message_data[1:5])[0]
        data = message_data[5:5+length] if len(message_data) >= 5+length else b''
        
        parse_time = (time.perf_counter() - start_time) * 1000
        self.parse_times.append(parse_time)
        
        return {
            'compression': compression,
            'length': length,
            'data_size': len(data),
            'parse_time_ms': parse_time
        }
    
    def generate_test_messages(self, count: int, protocol: str) -> List[Any]:
        """Generate test messages for parsing"""
        messages = []
        
        if protocol == 'HTTP/1.1':
            for i in range(count):
                message = f"""GET /api/v1/service/{i} HTTP/1.1
Host: example.com
User-Agent: LoadTester/1.0
Content-Type: application/json
Content-Length: 50
Authorization: Bearer token_{i}

{{"request_id": "{i}", "data": "test_payload_{i}"}}"""
                messages.append(message)
        
        elif protocol == 'gRPC':
            for i in range(count):
                # Simulate gRPC binary message
                data = json.dumps({"request_id": i, "data": f"grpc_payload_{i}"}).encode()
                length = len(data)
                message = bytes([0]) + struct.pack('>I', length) + data
                messages.append(message)
        
        return messages
    
    def benchmark_parsing(self, protocol: str, num_messages: int = 1000) -> Dict[str, float]:
        """Benchmark protocol parsing performance"""
        messages = self.generate_test_messages(num_messages, protocol)
        
        start_time = time.time()
        parsed_count = 0
        error_count = 0
        
        for message in messages:
            try:
                if protocol == 'HTTP/1.1':
                    result = self.parse_http_request(message)
                elif protocol == 'gRPC':
                    result = self.parse_grpc_message(message)
                
                if result:
                    parsed_count += 1
                else:
                    error_count += 1
            except Exception as e:
                error_count += 1
        
        total_time = time.time() - start_time
        
        return {
            'protocol': protocol,
            'total_time_ms': total_time * 1000,
            'messages_parsed': parsed_count,
            'parse_errors': error_count,
            'messages_per_second': parsed_count / total_time,
            'avg_parse_time_ms': statistics.mean(self.parse_times) if self.parse_times else 0,
            'max_parse_time_ms': max(self.parse_times) if self.parse_times else 0,
            'min_parse_time_ms': min(self.parse_times) if self.parse_times else 0
        }