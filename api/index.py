from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import json

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Sample data for demo (since we can't store files on Vercel)
DEMO_DATA = {
    "match_197c": {
        "job_id": "match_197c",
        "type": "combined_match",
        "result": {
            "sport_type": "squash",
            "match_info": {
                "total_duration_seconds": 526,
                "total_duration_formatted": "8m 46s",
                "total_games": 1
            },
            "squash_analytics": {
                "player1": {
                    "avg_t_dominance": 41,
                    "avg_scramble_score": 279.9,
                    "total_running_score": 93305.6,
                    "total_attack_score": 200,
                    "avg_rail_distance": 704.5,
                    "total_tight_rails": 0
                },
                "player2": {
                    "avg_t_dominance": 59,
                    "avg_scramble_score": 218.4,
                    "total_running_score": 88117.7,
                    "total_attack_score": 181,
                    "avg_rail_distance": 947.5,
                    "total_tight_rails": 0
                },
                "analysis": {
                    "t_dominance": {"summary": "Player 2 controlled the T more consistently across the match."}
                }
            }
        }
    }
}

@app.route('/')
def index():
    return render_template('customer/upload.html')

@app.route('/results/<job_id>')
def results(job_id):
    return render_template('customer/results.html', job_id=job_id)

@app.route('/match/<match_id>')
def match_results(match_id):
    return render_template('customer/results.html', job_id=match_id)

@app.route('/api/results/<job_id>')
def get_results_data(job_id):
    # Return demo data for Vercel deployment
    if job_id in DEMO_DATA:
        return jsonify(DEMO_DATA[job_id])
    return jsonify({"error": "Job not found", "job_id": job_id}), 404

@app.route('/api/recent-jobs')
def recent_jobs():
    return jsonify([])

# Vercel handler
def handler(request):
    return app(request)

# For local testing
if __name__ == '__main__':
    app.run(debug=True, port=5001)

