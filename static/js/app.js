// YOLO Video Detection Dashboard - Frontend JavaScript

let topObjectsChart = null;

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    loadVideos();
    
    // Setup modal close button
    const modal = document.getElementById('videoModal');
    const closeBtn = document.getElementsByClassName('close')[0];
    
    closeBtn.onclick = function() {
        modal.style.display = 'none';
    };
    
    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    };
});

// Load overall statistics
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        if (data.success) {
            const stats = data.stats;
            
            document.getElementById('totalVideos').textContent = stats.total_videos;
            document.getElementById('totalFrames').textContent = formatNumber(stats.total_frames);
            document.getElementById('totalDetections').textContent = formatNumber(stats.total_detections);
            document.getElementById('totalDuration').textContent = formatDuration(stats.total_duration);
            
            // Create top objects chart
            createTopObjectsChart(stats.all_classes);
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Load videos list
async function loadVideos() {
    try {
        const response = await fetch('/api/videos');
        const data = await response.json();
        
        if (data.success) {
            displayVideos(data.videos);
        }
    } catch (error) {
        console.error('Error loading videos:', error);
        document.getElementById('videosList').innerHTML = 
            '<div class="loading">Error loading videos. Please try again.</div>';
    }
}

// Display videos in the grid
function displayVideos(videos) {
    const videosList = document.getElementById('videosList');
    
    if (videos.length === 0) {
        videosList.innerHTML = '<div class="loading">No videos processed yet. Run your detection scripts first!</div>';
        return;
    }
    
    videosList.innerHTML = videos.map(video => createVideoCard(video)).join('');
    
    // Add click handlers
    videos.forEach(video => {
        const card = document.querySelector(`[data-video="${video.video_name}"]`);
        if (card) {
            card.addEventListener('click', () => showVideoDetails(video.video_name));
        }
    });
}

// Create video card HTML
function createVideoCard(video) {
    const topDetections = Object.entries(video.detection_summary)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([name, count]) => ({ name, count }));
    
    const topTags = topDetections.map(d => 
        `<span class="detection-tag">${d.name} (${formatNumber(d.count)})</span>`
    ).join('');
    
    return `
        <div class="video-card" data-video="${video.video_name}">
            <div class="video-card-header">
                <div class="video-name">${video.video_name}</div>
                ${video.has_annotated_video ? '<span class="video-badge">✓ Video</span>' : ''}
            </div>
            <div class="video-info">
                <div class="info-item">
                    <div class="info-label">Duration</div>
                    <div class="info-value">${formatDuration(video.duration_seconds)}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Frames</div>
                    <div class="info-value">${formatNumber(video.total_frames)}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">FPS</div>
                    <div class="info-value">${video.fps}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Detections</div>
                    <div class="info-value">${formatNumber(video.total_detections)}</div>
                </div>
            </div>
            <div class="detection-preview">
                <h4>Top Detections</h4>
                <div class="detection-tags">
                    ${topTags}
                </div>
            </div>
        </div>
    `;
}

// Show video details in modal
async function showVideoDetails(videoName) {
    const modal = document.getElementById('videoModal');
    const modalBody = document.getElementById('modalBody');
    
    modal.style.display = 'block';
    modalBody.innerHTML = '<div class="loading">Loading video details...</div>';
    
    try {
        const response = await fetch(`/api/video/${videoName}`);
        const data = await response.json();
        
        if (data.success) {
            displayVideoDetails(data.video);
        } else {
            modalBody.innerHTML = '<div class="loading">Error loading video details.</div>';
        }
    } catch (error) {
        console.error('Error loading video details:', error);
        modalBody.innerHTML = '<div class="loading">Error loading video details.</div>';
    }
}

// Display video details in modal
function displayVideoDetails(video) {
    const modalBody = document.getElementById('modalBody');
    
    // Sort detections by count
    const sortedDetections = Object.entries(video.detection_summary)
        .sort((a, b) => b[1] - a[1])
        .map(([name, count]) => ({ name, count }));
    
    const detectionItems = sortedDetections.map(d => `
        <div class="detection-item">
            <div class="detection-item-class">${d.name}</div>
            <div class="detection-item-count">${formatNumber(d.count)} detections</div>
        </div>
    `).join('');
    
    const videoPlayer = video.has_annotated_video ? `
        <div class="video-player-container">
            <video controls>
                <source src="/videos/${video.video_name}_annotated.mp4" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
    ` : '<p style="color: var(--text-secondary);">No annotated video available.</p>';
    
    modalBody.innerHTML = `
        <div class="modal-header">
            <h2>${video.video_name}</h2>
            <p style="color: var(--text-secondary);">Detailed detection results</p>
        </div>
        
        <div class="video-info" style="grid-template-columns: repeat(4, 1fr); margin-bottom: 30px;">
            <div class="info-item">
                <div class="info-label">Duration</div>
                <div class="info-value">${formatDuration(video.duration_seconds)}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Total Frames</div>
                <div class="info-value">${formatNumber(video.total_frames)}</div>
            </div>
            <div class="info-item">
                <div class="info-label">FPS</div>
                <div class="info-value">${video.fps}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Total Detections</div>
                <div class="info-value">${formatNumber(video.total_detections)}</div>
            </div>
        </div>
        
        ${videoPlayer}
        
        <div class="detection-timeline">
            <h3 style="margin-bottom: 15px;">All Detected Objects</h3>
            <div class="detection-list">
                ${detectionItems}
            </div>
        </div>
        
        ${video.summary_text ? `
            <div style="margin-top: 30px; padding: 20px; background: rgba(99, 102, 241, 0.1); border-radius: 8px; border: 1px solid var(--primary-color);">
                <h3 style="margin-bottom: 15px;">Detection Summary</h3>
                <pre style="color: var(--text-secondary); white-space: pre-wrap; font-family: inherit;">${video.summary_text}</pre>
            </div>
        ` : ''}
    `;
    
    // Create timeline chart if available
    if (video.timeline && video.timeline.length > 0) {
        createTimelineChart(video.timeline, video.video_name);
    }
}

// Create top objects chart
function createTopObjectsChart(allClasses) {
    const ctx = document.getElementById('topObjectsChart');
    if (!ctx) return;
    
    // Get top 10 objects
    const sorted = Object.entries(allClasses)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);
    
    const labels = sorted.map(([name]) => name);
    const data = sorted.map(([, count]) => count);
    
    if (topObjectsChart) {
        topObjectsChart.destroy();
    }
    
    topObjectsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Detection Count',
                data: data,
                backgroundColor: 'rgba(99, 102, 241, 0.6)',
                borderColor: 'rgba(99, 102, 241, 1)',
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return formatNumber(context.parsed.y) + ' detections';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#cbd5e1',
                        callback: function(value) {
                            return formatNumber(value);
                        }
                    },
                    grid: {
                        color: 'rgba(203, 213, 225, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#cbd5e1'
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// Create timeline chart
function createTimelineChart(timeline, videoName) {
    // Find a container or create one
    let container = document.querySelector('.timeline-chart-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'timeline-chart-container';
        const timelineDiv = document.querySelector('.detection-timeline');
        if (timelineDiv) {
            timelineDiv.insertBefore(container, timelineDiv.firstChild);
        }
    }
    
    // Create canvas
    const canvas = document.createElement('canvas');
    container.innerHTML = '';
    container.appendChild(canvas);
    
    // Extract data
    const timestamps = timeline.map(t => t.timestamp);
    const totalObjects = timeline.map(t => t.total_objects);
    
    new Chart(canvas, {
        type: 'line',
        data: {
            labels: timestamps.map(t => formatDuration(t)),
            datasets: [{
                label: 'Objects Detected',
                data: totalObjects,
                borderColor: 'rgba(139, 92, 246, 1)',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#cbd5e1'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#cbd5e1'
                    },
                    grid: {
                        color: 'rgba(203, 213, 225, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#cbd5e1',
                        maxTicksLimit: 10
                    },
                    grid: {
                        color: 'rgba(203, 213, 225, 0.1)'
                    }
                }
            }
        }
    });
}

// Utility functions
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toLocaleString();
}

function formatDuration(seconds) {
    if (seconds < 60) {
        return seconds.toFixed(1) + 's';
    } else if (seconds < 3600) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}m ${secs}s`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${mins}m`;
    }
}







