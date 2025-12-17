#!/usr/bin/env python3
"""
Script to re-run squash analytics on existing detection data.
"""

import json
import sys
from squash_analytics import analyze_squash_match

def rerun_analytics(job_id):
    """Re-run analytics for a given job ID."""
    
    # Find the detection data file
    import os
    results_dir = f"customer_results/{job_id}"
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return False
    
    # Find detection data file
    detection_file = None
    for f in os.listdir(results_dir):
        if f.endswith("_detection_data.json"):
            detection_file = os.path.join(results_dir, f)
            break
    
    if not detection_file:
        print(f"Error: No detection data file found in {results_dir}")
        return False
    
    print(f"Loading detection data from: {detection_file}")
    
    # Load detection data
    with open(detection_file, 'r') as f:
        detection_data = json.load(f)
    
    print(f"Loaded {len(detection_data.get('detections', []))} frames")
    
    # Run analytics
    print("Running squash analytics...")
    analytics = analyze_squash_match(detection_data)
    
    # Check what's in the new analytics
    print("\n=== Analytics Summary ===")
    print(f"Match duration: {analytics.get('match_info', {}).get('duration_seconds', 0):.1f}s")
    print(f"Player 1 T-Dominance: {analytics.get('player1', {}).get('t_dominance', 0)}%")
    print(f"Player 2 T-Dominance: {analytics.get('player2', {}).get('t_dominance', 0)}%")
    
    # Check for new analytics
    if 'zone_dwell_time' in analytics:
        print("\n✓ Zone Dwell Time analysis included:")
        zone_data = analytics['zone_dwell_time']
        p1_zones = zone_data.get('player1', {})
        p2_zones = zone_data.get('player2', {})
        print(f"  Player 1 - Front: {p1_zones.get('front', 0)}%, Mid: {p1_zones.get('mid', 0)}%, Back: {p1_zones.get('back', 0)}%")
        print(f"  Player 2 - Front: {p2_zones.get('front', 0)}%, Mid: {p2_zones.get('mid', 0)}%, Back: {p2_zones.get('back', 0)}%")
    else:
        print("\n✗ Zone Dwell Time NOT found in analytics")
    
    if 'performance_decay' in analytics:
        print("\n✓ Performance Decay analysis included:")
        decay_data = analytics['performance_decay']
        p1_decay = decay_data.get('player1', {})
        p2_decay = decay_data.get('player2', {})
        print(f"  Player 1 - T-Control change: {p1_decay.get('t_dominance_change', 0):+.1f}%")
        print(f"  Player 2 - T-Control change: {p2_decay.get('t_dominance_change', 0):+.1f}%")
    else:
        print("\n✗ Performance Decay NOT found in analytics")
    
    # Save updated analytics
    analytics_file = detection_file.replace("_detection_data.json", "_squash_analytics.json")
    print(f"\nSaving updated analytics to: {analytics_file}")
    
    with open(analytics_file, 'w') as f:
        json.dump(analytics, f, indent=2)
    
    print("Done!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rerun_analytics.py <job_id>")
        sys.exit(1)
    
    job_id = sys.argv[1]
    success = rerun_analytics(job_id)
    sys.exit(0 if success else 1)

