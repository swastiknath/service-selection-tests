# Instance Selection Algorithms: Comparative Benchmarks
#### Authors: Swastik Nath, Anushree Chakraborty, Soumyajit Pal

We perform comparative analysis of several performance metrics including but not limited to memory overheads, CPU utilization, Average Response Time, System Througput of certain Instance Selection Algorithms used in Microservices Mesh architecutures in different deployment scenarios such as with diffrent sidecar proxies/ proxyless models and hybrid models.
We also test the performance aspects of Centralized and Distributed Control Planes with Google Kubernetes Engine (GKE) with In-Cluster versus Google Managed Control Planes. 


## Algorithm Performance Summary
### Best Performing Algorithms
- **Highest Throughput**: Adaptive Context-based instance selection (2533.24 RPS in high_load with linkerd)
- **Lowest Response Time**: Adaptive Context-based instance selection (20.93 ms in low_load with no_mesh)
- **Lowest CPU Usage**: Stochastic Quality of Service parameters (14.72% in low_load with linkerd)
- **Lowest Memory Usage**: Stochastic Quality of Service parameters (129.34 MB in low_load with no_mesh)

### Algorithm Comparison (Average Performance)
| Algorithm | Throughput (RPS) | Response Time (ms) | Success Rate | CPU Usage (%) | Memory (MB) | Selection Time (ms) |
|-----------|------------------|-------------------|--------------|---------------|-------------|-------------------|
| Adaptive Context-based instance selection | 1507.17 | 52.0 | 0.990 | 59.85 | 258.51 | 3.26 |
| Fuzzy Preference Relations | 1336.76 | 61.82 | 0.990 | 50.02 | 243.15 | 2.51 |
| Stochastic Quality of Service parameters | 1107.34 | 70.39 | 0.990 | 42.2 | 222.12 | 1.8 |

### Service Mesh Performance Impact
| Mesh Type | Throughput (RPS) | Response Time (ms) | CPU Usage (%) | Memory (MB) |
|-----------|------------------|-------------------|---------------|-------------|
| consul_connect | 1302.65 | 64.87 | 52.53 | 244.9 |
| istio | 1307.94 | 59.69 | 56.11 | 261.75 |
| linkerd | 1435.35 | 55.91 | 47.93 | 239.53 |
| no_mesh | 1222.41 | 65.15 | 46.2 | 218.86 |

## IPC Performance Results
### Shared Memory
- Messages per Second: 46254.84
- Latency per Message: 0.0405 ms
- Total Time: 216.19 ms

### Message Queue
- Messages per Second: 11460.07
- Latency per Message: 0.1334 ms
- Total Time: 872.59 ms

### Socket
- Messages per Second: 8902.00
- Latency per Message: 0.2677 ms
- Total Time: 1123.34 ms

## Protocol Parsing Performance
### HTTP
- Messages per Second: 26960.90
- Average Parse Time: 0.0668 ms
- Parse Errors: 9

### gRPC
- Messages per Second: 31708.58
- Average Parse Time: 0.0444 ms
- Parse Errors: 5

### TCP
- Messages per Second: 45265.36
- Average Parse Time: 0.0264 ms
- Parse Errors: 2

### WebSocket
- Messages per Second: 18161.79
- Average Parse Time: 0.1066 ms
- Parse Errors: 21

## Notification System Performance
- Notifications per Second: 2309.43
- Average Notification Time: 0.8045 ms
- Subscribers: 231

## Recommendations
- **For production use**: Stochastic Quality of Service parameters algorithm shows the best balance of performance and resource efficiency
- **Service Mesh**: linkerd provides the best overall performance
- Monitor CPU and memory usage in production environments
- Consider load balancing algorithms based on specific use case requirements
- Implement proper monitoring and alerting for instance selection performance
