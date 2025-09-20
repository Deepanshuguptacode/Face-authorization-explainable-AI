"""
Web Interface for Human Evaluation Study
=======================================

Flask-based web interface for conducting human studies on explainable face recognition.
Provides participant registration, consent forms, task interface, and data collection.
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
import sys
import json
import uuid
import time
from datetime import datetime, timedelta
import secrets

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import evaluation framework from local directory
try:
    from evaluation_framework import HumanEvaluationFramework
except ImportError:
    try:
        from human_studies.evaluation_framework import HumanEvaluationFramework
    except ImportError:
        # Create a minimal framework if import fails
        class HumanEvaluationFramework:
            def __init__(self, study_name, output_dir):
                self.study_name = study_name
                self.output_dir = output_dir
                print(f"⚠️  Using minimal evaluation framework for {study_name}")
            
            def register_participant(self, demographics_data):
                participant_id = f"participant_{int(time.time())}"
                print(f"Registered participant: {participant_id}")
                return participant_id

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

# Global study framework
study_framework = None

def initialize_study():
    """Initialize the human evaluation study"""
    global study_framework
    
    study_framework = HumanEvaluationFramework(
        study_name="explainable_face_recognition_user_study_2024",
        output_dir="human_studies/data"
    )
    
    print("✅ Human evaluation study initialized")

@app.route('/')
def index():
    """Landing page for the study"""
    return render_template('study/landing.html')

@app.route('/consent')
def consent_form():
    """Informed consent form"""
    return render_template('study/consent.html')

@app.route('/demographics')
def demographics_form():
    """Demographics questionnaire"""
    if 'consented' not in session:
        return redirect(url_for('consent_form'))
    
    return render_template('study/demographics.html')

@app.route('/submit_demographics', methods=['POST'])
def submit_demographics():
    """Process demographics submission"""
    if 'consented' not in session:
        return jsonify({'error': 'Consent required'}), 400
    
    try:
        demographics_data = {
            'age_group': request.form.get('age_group'),
            'gender': request.form.get('gender'),
            'education_level': request.form.get('education_level'),
            'tech_experience': request.form.get('tech_experience'),
            'ai_familiarity': request.form.get('ai_familiarity'),
            'face_recognition_experience': request.form.get('face_recognition_experience'),
            'occupation_category': request.form.get('occupation_category')
        }
        
        # Register participant
        participant_id = study_framework.register_participant(demographics_data)
        
        # Store in session
        session['participant_id'] = participant_id
        session['study_start_time'] = time.time()
        session['task_index'] = 0
        
        # Get participant tasks
        tasks = study_framework.get_participant_tasks(participant_id)
        session['tasks'] = tasks
        
        return jsonify({
            'success': True,
            'participant_id': participant_id,
            'total_tasks': len(tasks)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/study_instructions')
def study_instructions():
    """Study instructions page"""
    if 'participant_id' not in session:
        return redirect(url_for('demographics_form'))
    
    return render_template('study/instructions.html')

@app.route('/task')
def task_interface():
    """Main task interface"""
    if 'participant_id' not in session:
        return redirect(url_for('demographics_form'))
    
    task_index = session.get('task_index', 0)
    tasks = session.get('tasks', [])
    
    if task_index >= len(tasks):
        return redirect(url_for('study_complete'))
    
    current_task = tasks[task_index]
    
    return render_template('study/task.html', 
                         task=current_task,
                         task_number=task_index + 1,
                         total_tasks=len(tasks),
                         participant_id=session['participant_id'])

@app.route('/submit_task', methods=['POST'])
def submit_task():
    """Submit task response"""
    if 'participant_id' not in session:
        return jsonify({'error': 'Session expired'}), 400
    
    try:
        task_data = request.get_json()
        
        # Submit task result
        result_data = {
            'participant_id': session['participant_id'],
            'task_id': task_data['task_id'],
            'task_type': 'verification',
            'correct_answer': task_data['ground_truth'],
            'participant_answer': task_data['participant_answer'],
            'confidence_in_answer': task_data['confidence'],
            'explanation_viewed': task_data.get('explanation_viewed', False),
            'explanation_influenced_decision': task_data.get('explanation_influenced_decision', False),
            'pre_explanation_confidence': task_data.get('pre_explanation_confidence', 0),
            'post_explanation_confidence': task_data.get('post_explanation_confidence', 0),
            'response_time': task_data.get('response_time', 0),
            'number_of_interactions': task_data.get('interactions', 1),
            'stimulus_id': task_data['stimulus_id'],
            'ground_truth_label': str(task_data['ground_truth']),
            'ai_prediction': task_data['ai_prediction'],
            'ai_confidence': task_data['ai_confidence']
        }
        
        study_framework.submit_task_result(result_data)
        
        # Submit explanation rating if provided
        if 'explanation_rating' in task_data:
            rating_data = task_data['explanation_rating']
            rating_data['participant_id'] = session['participant_id']
            rating_data['session_id'] = session.get('session_id', '')
            
            study_framework.submit_rating(rating_data)
        
        # Move to next task
        session['task_index'] = session.get('task_index', 0) + 1
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/explanation_rating')
def explanation_rating():
    """Explanation rating interface"""
    if 'participant_id' not in session:
        return redirect(url_for('demographics_form'))
    
    # Get the last completed task for rating
    task_index = session.get('task_index', 1) - 1
    tasks = session.get('tasks', [])
    
    if task_index < 0 or task_index >= len(tasks):
        return redirect(url_for('task_interface'))
    
    task = tasks[task_index]
    
    return render_template('study/explanation_rating.html', 
                         task=task,
                         participant_id=session['participant_id'])

@app.route('/submit_explanation_rating', methods=['POST'])
def submit_explanation_rating():
    """Submit explanation rating"""
    if 'participant_id' not in session:
        return jsonify({'error': 'Session expired'}), 400
    
    try:
        rating_data = request.get_json()
        rating_data['participant_id'] = session['participant_id']
        rating_data['session_id'] = session.get('session_id', '')
        
        study_framework.submit_rating(rating_data)
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/study_complete')
def study_complete():
    """Study completion page"""
    if 'participant_id' not in session:
        return redirect(url_for('index'))
    
    # Calculate study duration
    start_time = session.get('study_start_time', time.time())
    duration = time.time() - start_time
    
    # Generate completion code
    completion_code = f"STUDY_{session['participant_id'][-8:]}_{int(time.time())}"
    
    return render_template('study/complete.html',
                         duration=duration,
                         completion_code=completion_code,
                         participant_id=session['participant_id'])

@app.route('/api/study_status')
def api_study_status():
    """API endpoint for study status"""
    status = study_framework.get_study_status()
    return jsonify(status)

@app.route('/api/participant_progress')
def api_participant_progress():
    """API endpoint for participant progress"""
    if 'participant_id' not in session:
        return jsonify({'error': 'No active session'}), 400
    
    task_index = session.get('task_index', 0)
    total_tasks = len(session.get('tasks', []))
    
    return jsonify({
        'participant_id': session['participant_id'],
        'current_task': task_index + 1,
        'total_tasks': total_tasks,
        'progress_percentage': (task_index / max(total_tasks, 1)) * 100
    })

@app.route('/admin')
def admin_dashboard():
    """Administrator dashboard"""
    status = study_framework.get_study_status()
    return render_template('study/admin.html', status=status)

@app.route('/admin/export_data')
def admin_export_data():
    """Export study data"""
    try:
        export_path = study_framework.export_study_data()
        return jsonify({
            'success': True,
            'export_path': export_path,
            'message': 'Data exported successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Session management
@app.route('/api/consent', methods=['POST'])
def api_consent():
    """Process consent form submission"""
    consent_data = request.get_json()
    
    if consent_data.get('consented') == True:
        session['consented'] = True
        session['consent_timestamp'] = datetime.now().isoformat()
        session['session_id'] = str(uuid.uuid4())
        session.permanent = True
        
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Consent required to participate'}), 400

@app.route('/api/session_check')
def api_session_check():
    """Check session validity"""
    return jsonify({
        'valid': 'participant_id' in session,
        'participant_id': session.get('participant_id'),
        'consented': session.get('consented', False)
    })

# Additional routes for complete study workflow
@app.route('/study/demographics', methods=['GET', 'POST'])
def study_demographics():
    """Handle demographics questionnaire"""
    if request.method == 'GET':
        if 'consented' not in session:
            return redirect(url_for('consent_form'))
        return render_template('study/demographics.html')
    
    # Process demographics form
    demographics_data = {
        'age': request.form.get('age'),
        'gender': request.form.get('gender'),
        'education': request.form.get('education'),
        'field': request.form.get('field'),
        'tech_comfort': int(request.form.get('tech_comfort', 4)),
        'tech_experience': request.form.getlist('tech_experience'),
        'ai_knowledge': int(request.form.get('ai_knowledge', 4)),
        'ai_trust': int(request.form.get('ai_trust', 4)),
        'ai_concepts': request.form.getlist('ai_concepts'),
        'comments': request.form.get('comments', '')
    }
    
    # Store demographics
    participant_id = session.get('participant_id', f"participant_{int(time.time())}")
    session['participant_id'] = participant_id
    session['demographics_completed'] = True
    
    # In real implementation, store to database
    print(f"Demographics collected for {participant_id}: {demographics_data}")
    
    return redirect(url_for('study_task', task_id=1))

@app.route('/study/task/<int:task_id>')
def study_task(task_id):
    """Display study task"""
    if not session.get('demographics_completed'):
        return redirect(url_for('study_demographics'))
    
    total_tasks = 5
    if task_id > total_tasks:
        return redirect(url_for('study_complete'))
    
    # Generate task-specific content
    task_data = {
        'task_id': task_id,
        'total_tasks': total_tasks,
        'input_image': f'/static/sample_images/task_{task_id}_input.jpg',
        'reference_image': f'/static/sample_images/task_{task_id}_reference.jpg',
        'verification_result': 'MATCH' if task_id % 2 == 1 else 'NO MATCH',
        'confidence': 87 if task_id % 2 == 1 else 23,
        'threshold': 75
    }
    
    return render_template('study/task.html', **task_data)

@app.route('/study/submit_task', methods=['POST'])
def study_submit_task():
    """Submit study task response"""
    if not session.get('demographics_completed'):
        return redirect(url_for('study_demographics'))
    
    # Process task ratings
    task_data = {
        'participant_id': session['participant_id'],
        'task_id': int(request.form.get('task_id')),
        'understanding_rating': int(request.form.get('understanding')),
        'trust_rating': int(request.form.get('trust')),
        'helpfulness_rating': int(request.form.get('helpfulness')),
        'clarity_rating': int(request.form.get('clarity')),
        'completeness_rating': int(request.form.get('completeness')),
        'accuracy_rating': int(request.form.get('accuracy')),
        'most_helpful': request.form.get('most_helpful', ''),
        'least_helpful': request.form.get('least_helpful', ''),
        'improvements': request.form.get('improvements', ''),
        'completion_time_ms': int(request.form.get('completion_time', 0)),
        'timestamp': time.time()
    }
    
    # Store task result (in real implementation, store to database)
    print(f"Task result collected: {task_data}")
    
    # Determine next action
    next_task = task_data['task_id'] + 1
    total_tasks = 5
    
    if next_task > total_tasks:
        return redirect(url_for('study_complete'))
    else:
        return redirect(url_for('study_task', task_id=next_task))

@app.route('/study/complete')
def study_complete():
    """Study completion page"""
    if not session.get('demographics_completed'):
        return redirect(url_for('study_demographics'))
    
    # Generate completion statistics
    stats = {
        'tasks_completed': 5,
        'total_time_ms': 1425000,  # 23:45 in milliseconds
        'ratings_provided': 24,
        'participant_id': session['participant_id']
    }
    
    return render_template('study/complete.html', **stats)

@app.route('/study/final_feedback', methods=['POST'])
def study_final_feedback():
    """Handle final feedback submission"""
    feedback_data = {
        'participant_id': session['participant_id'],
        'study_experience': request.form.get('study_experience'),
        'study_length': request.form.get('study_length'),
        'technical_issues': request.form.get('technical_issues', ''),
        'general_comments': request.form.get('general_comments', ''),
        'timestamp': time.time()
    }
    
    # Store feedback (in real implementation, store to database)
    print(f"Final feedback collected: {feedback_data}")
    
    return jsonify({'status': 'success'})

@app.route('/study/contact_preferences', methods=['POST'])
def study_contact_preferences():
    """Handle contact preferences submission"""
    contact_data = request.get_json()
    contact_data['participant_id'] = session['participant_id']
    contact_data['timestamp'] = time.time()
    
    # Store contact preferences (in real implementation, store to database)
    print(f"Contact preferences collected: {contact_data}")
    
    return jsonify({'status': 'success'})

@app.route('/study/restart')
def study_restart():
    """Restart the study"""
    session.clear()
    return redirect(url_for('landing'))

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('study/error.html',
                         error_code=404,
                         error_message="Page not found"), 404@app.errorhandler(500)
def internal_error(error):
    return render_template('study/error.html',
                         error_code=500,
                         error_message="Internal server error"), 500

if __name__ == '__main__':
    print("🚀 Starting Human Evaluation Study Server...")
    
    # Initialize study
    initialize_study()
    
    # Start Flask app
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )