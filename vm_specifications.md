# Google Cloud VM Specifications for Video Analysis

## Recommended Configuration

### **Primary Recommendation (Balanced Performance/Cost)**
- **Machine Type**: `n1-standard-4`
- **vCPUs**: 4
- **Memory**: 15 GB RAM
- **GPU**: 1x NVIDIA T4 (16 GB VRAM)
- **Storage**: 50 GB SSD persistent disk
- **Operating System**: Ubuntu 20.04 LTS
- **Estimated Cost**: $0.50-0.80/hour

### **High-Performance Option (For Large Batches)**
- **Machine Type**: `n1-standard-8`
- **vCPUs**: 8
- **Memory**: 30 GB RAM
- **GPU**: 1x NVIDIA T4 (16 GB VRAM) or 1x NVIDIA V100 (32 GB VRAM)
- **Storage**: 100 GB SSD persistent disk
- **Operating System**: Ubuntu 20.04 LTS
- **Estimated Cost**: $0.80-1.20/hour (T4) or $2.50-3.50/hour (V100)

### **Budget Option (CPU-Only)**
- **Machine Type**: `n1-standard-4`
- **vCPUs**: 4
- **Memory**: 15 GB RAM
- **GPU**: None (CPU processing)
- **Storage**: 50 GB SSD persistent disk
- **Operating System**: Ubuntu 20.04 LTS
- **Estimated Cost**: $0.20-0.30/hour
- **Performance**: 3-5x slower than GPU option

## Technical Justification

### **CPU Requirements**
- **4+ vCPUs**: YOLO inference is CPU-intensive for preprocessing
- **High-frequency cores**: n1-standard provides better single-thread performance
- **Multi-threading**: Video processing benefits from parallel frame processing

### **Memory Requirements**
- **15+ GB RAM**: 
  - YOLO model: ~6 MB
  - Video frames in memory: ~100-500 MB per video
  - OpenCV buffers: ~200-500 MB
  - System overhead: ~2-4 GB
  - Processing buffers: ~2-5 GB

### **GPU Requirements**
- **NVIDIA T4**: 
  - 16 GB VRAM (sufficient for 4K video processing)
  - Tensor cores for AI acceleration
  - 3-5x faster inference than CPU
  - Cost-effective for video processing workloads

### **Storage Requirements**
- **SSD persistent disk**: Fast I/O for video file access
- **50+ GB**: 
  - OS and dependencies: ~10 GB
  - YOLO models: ~100 MB
  - Video files: ~5-20 GB (depending on batch size)
  - Output files: ~10-30 GB (annotated videos + JSON data)

## Performance Expectations

### **Processing Speed**
- **GPU (T4)**: 15-25 FPS processing speed
- **CPU-only**: 3-8 FPS processing speed
- **Example**: 10-minute video (18,000 frames at 30 FPS)
  - GPU: ~12-20 minutes processing time
  - CPU: ~45-90 minutes processing time

### **Batch Processing Capacity**
- **Single video**: Any length (memory permitting)
- **Multiple videos**: 5-10 videos simultaneously (depending on resolution)
- **Storage**: Can process 50-100 GB of video files per session

## Cost Analysis

### **Hourly Costs (Approximate)**
- **n1-standard-4 + T4**: $0.50-0.80/hour
- **n1-standard-8 + T4**: $0.80-1.20/hour
- **n1-standard-4 (CPU-only)**: $0.20-0.30/hour

### **Processing Cost Examples**
- **10-minute video**: $0.10-0.20 (GPU) vs $0.50-1.00 (CPU)
- **1-hour video**: $0.50-1.00 (GPU) vs $2.50-5.00 (CPU)
- **Batch of 10 videos**: $2.00-4.00 (GPU) vs $10.00-20.00 (CPU)

### **Cost Optimization Strategies**
1. **Preemptible instances**: 60-80% cost savings
2. **Spot instances**: Up to 90% cost savings
3. **Auto-shutdown**: Stop VM when not processing
4. **Right-sizing**: Start with smaller instance, scale up if needed

## Alternative Configurations

### **Development/Testing**
- **Machine Type**: `e2-standard-2`
- **vCPUs**: 2
- **Memory**: 8 GB RAM
- **GPU**: None
- **Cost**: $0.10-0.15/hour

### **Production/High-Volume**
- **Machine Type**: `n1-standard-16`
- **vCPUs**: 16
- **Memory**: 60 GB RAM
- **GPU**: 1x NVIDIA V100 (32 GB VRAM)
- **Cost**: $3.00-4.00/hour

## Network and Security

### **Network Configuration**
- **Bandwidth**: 10 Gbps (sufficient for video upload/download)
- **Firewall**: Allow SSH (port 22) and HTTP/HTTPS (ports 80/443)
- **VPC**: Default network configuration acceptable

### **Security Considerations**
- **SSH key authentication**: Recommended over password
- **Firewall rules**: Restrict access to specific IP ranges
- **Disk encryption**: Enabled by default on GCP
- **Access logging**: Cloud Audit Logs enabled

## Monitoring and Maintenance

### **Performance Monitoring**
- **GPU utilization**: `nvidia-smi` for GPU usage
- **CPU/Memory**: Standard system monitoring
- **Disk I/O**: Monitor for bottlenecks
- **Network**: Track upload/download speeds

### **Maintenance Tasks**
- **Regular updates**: OS and security patches
- **Model updates**: YOLO model version updates
- **Storage cleanup**: Remove processed videos to save space
- **Cost monitoring**: Set up billing alerts

## Implementation Timeline

### **Setup Phase (1-2 hours)**
1. VM creation and configuration
2. Software installation and dependencies
3. Model download and verification
4. Test processing with sample video

### **Production Phase (Ongoing)**
1. Video upload and processing
2. Results download and analysis
3. VM shutdown when not in use
4. Cost monitoring and optimization

## Risk Mitigation

### **Technical Risks**
- **GPU availability**: T4 GPUs may have limited availability
- **Processing failures**: Implement retry logic and error handling
- **Storage limits**: Monitor disk usage and implement cleanup

### **Cost Risks**
- **Runaway costs**: Set up billing alerts and auto-shutdown
- **Inefficient processing**: Monitor performance metrics
- **Forgotten instances**: Implement automated shutdown policies

## Success Metrics

### **Performance KPIs**
- **Processing speed**: Target 20+ FPS with GPU
- **Batch efficiency**: Process 5+ videos simultaneously
- **Error rate**: <1% processing failures
- **Cost efficiency**: <$0.50 per hour of video processed

### **Business KPIs**
- **Time savings**: 3-5x faster than local processing
- **Scalability**: Handle 10x more videos than local setup
- **Reliability**: 99%+ uptime during processing
- **Cost effectiveness**: 50%+ cost savings vs. local GPU setup

