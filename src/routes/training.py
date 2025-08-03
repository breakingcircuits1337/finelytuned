from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import threading
from src.services.training_service import TrainingService

training_bp = Blueprint('training', __name__)
training_service = TrainingService()

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'txt', 'csv', 'jsonl', 'json'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@training_bp.route('/upload', methods=['POST'])
def upload_file():
    """Upload dataset file for training"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = current_app.config.get('UPLOAD_FOLDER', 'uploads')
            os.makedirs(upload_path, exist_ok=True)
            file_path = os.path.join(upload_path, filename)
            file.save(file_path)
            file_size = os.path.getsize(file_path)
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename,
                'file_size': file_size,
                'file_path': file_path
            }), 200

        return jsonify({'error': 'File type not allowed'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@training_bp.route('/estimate', methods=['POST'])
def estimate_training_time():
    """Estimate training time based on dataset size and device"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        device = data.get('device', 'cpu')
        model_name = data.get('model_name', 'gpt2')

        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 400

        estimation = training_service.estimate_training_time(file_path, device, model_name)
        return jsonify(estimation), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@training_bp.route('/start', methods=['POST'])
def start_training():
    """Start the fine-tuning process"""
    try:
        data = request.get_json()
        if training_service.is_training():
            return jsonify({'error': 'Training already in progress'}), 400

        config = {
            'file_path': data.get('file_path'),
            'device': data.get('device', 'cpu'),
            'model_name': data.get('model_name', 'gpt2'),
            'epochs': data.get('epochs', 3),
            'learning_rate': data.get('learning_rate', 5e-5),
            'batch_size': data.get('batch_size', 4),
            'system_prompt': data.get('system_prompt'),
            'pasted_text': data.get('pasted_text')
        }

        training_thread = threading.Thread(
            target=training_service.start_training,
            args=(config,)
        )
        training_thread.daemon = True
        training_thread.start()

        return jsonify({'message': 'Training started successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@training_bp.route('/progress', methods=['GET'])
def get_training_progress():
    """Get current training progress"""
    try:
        progress = training_service.get_progress()
        return jsonify(progress), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@training_bp.route('/stop', methods=['POST'])
def stop_training():
    """Stop the current training process"""
    try:
        training_service.stop_training()
        return jsonify({'message': 'Training stopped'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@training_bp.route('/models', methods=['GET'])
def get_available_models():
    """Get list of available pre-trained models"""
    models = [
        {'name': 'gpt2', 'description': 'GPT-2 Small (124M parameters)'},
        {'name': 'gpt2-medium', 'description': 'GPT-2 Medium (355M parameters)'},
        {'name': 'distilgpt2', 'description': 'DistilGPT-2 (82M parameters)'},
        {'name': 'microsoft/DialoGPT-small', 'description': 'DialoGPT Small (117M parameters)'}
    ]
    return jsonify(models), 200