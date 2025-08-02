import os
import json
import time
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import pandas as pd
from threading import Lock
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self):
        self.training_lock = Lock()
        self.is_training_flag = False
        self.progress_data = {
            'status': 'idle',
            'progress': 0,
            'current_epoch': 0,
            'total_epochs': 0,
            'loss': 0.0,
            'estimated_time_remaining': 0,
            'start_time': None,
            'message': 'Ready to start training'
        }
        self.should_stop = False
        
    def is_training(self):
        return self.is_training_flag
    
    def get_progress(self):
        with self.training_lock:
            return self.progress_data.copy()
    
    def update_progress(self, **kwargs):
        with self.training_lock:
            self.progress_data.update(kwargs)
    
    def estimate_training_time(self, file_path, device, model_name):
        """Estimate training time based on file size and device"""
        try:
            file_size = os.path.getsize(file_path)
            
            # Read file to count samples
            sample_count = self._count_samples(file_path)
            
            # Base estimation factors (in seconds per sample)
            base_factors = {
                'cpu': {
                    'gpt2': 2.0,
                    'gpt2-medium': 4.0,
                    'distilgpt2': 1.5,
                    'microsoft/DialoGPT-small': 2.2
                },
                'gpu': {
                    'gpt2': 0.3,
                    'gpt2-medium': 0.6,
                    'distilgpt2': 0.2,
                    'microsoft/DialoGPT-small': 0.35
                }
            }
            
            # Get factor for the specific model and device
            factor = base_factors.get(device, base_factors['cpu']).get(model_name, 2.0)
            
            # Estimate total time (3 epochs by default)
            epochs = 3
            estimated_seconds = sample_count * factor * epochs
            
            # Convert to hours and minutes
            hours = int(estimated_seconds // 3600)
            minutes = int((estimated_seconds % 3600) // 60)
            
            return {
                'sample_count': sample_count,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'estimated_hours': hours,
                'estimated_minutes': minutes,
                'estimated_total_seconds': int(estimated_seconds),
                'device': device,
                'model': model_name
            }
            
        except Exception as e:
            logger.error(f"Error estimating training time: {e}")
            return {'error': str(e)}
    
    def _count_samples(self, file_path):
        """Count the number of training samples in the file"""
        try:
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Count non-empty lines
                    return len([line for line in lines if line.strip()])
            
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                return len(df)
            
            elif file_path.endswith(('.jsonl', '.json')):
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.endswith('.jsonl'):
                        lines = f.readlines()
                        return len([line for line in lines if line.strip()])
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            return len(data)
                        else:
                            return 1
            
            return 100  # Default fallback
            
        except Exception as e:
            logger.error(f"Error counting samples: {e}")
            return 100  # Default fallback
    
    def _prepare_dataset(self, file_path, tokenizer, system_prompt=None, pasted_text=None):
        """Prepare dataset for training"""
        try:
            texts = []
            
            if pasted_text:
                # Use pasted text directly
                texts = [pasted_text]
            elif system_prompt:
                # Use system prompt
                texts = [system_prompt]
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f.readlines() if line.strip()]
            
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                # Assume the first text column contains the training data
                text_column = df.select_dtypes(include=['object']).columns[0]
                texts = df[text_column].dropna().tolist()
            
            elif file_path.endswith(('.jsonl', '.json')):
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.endswith('.jsonl'):
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                if 'text' in data:
                                    texts.append(data['text'])
                                elif isinstance(data, str):
                                    texts.append(data)
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'text' in item:
                                    texts.append(item['text'])
                                elif isinstance(item, str):
                                    texts.append(item)
                        elif isinstance(data, dict) and 'text' in data:
                            texts.append(data['text'])
            
            # Tokenize the texts
            tokenized_texts = []
            for text in texts:
                tokens = tokenizer(
                    text,
                    truncation=True,
                    padding=False,
                    max_length=512,
                    return_tensors=None
                )
                tokenized_texts.append(tokens)
            
            # Create dataset
            dataset = Dataset.from_list(tokenized_texts)
            return dataset
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise e
    
    def start_training(self, config):
        """Start the fine-tuning process"""
        try:
            self.is_training_flag = True
            self.should_stop = False
            
            self.update_progress(
                status='initializing',
                progress=0,
                message='Initializing training...',
                start_time=time.time()
            )
            
            # Load model and tokenizer
            model_name = config['model_name']
            device = config['device']
            
            logger.info(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Move model to device
            if device == 'gpu' and torch.cuda.is_available():
                model = model.cuda()
                device_str = 'cuda'
            else:
                device_str = 'cpu'
            
            self.update_progress(
                status='preparing_data',
                progress=20,
                message='Preparing dataset...'
            )
            
            # Prepare dataset
            dataset = self._prepare_dataset(
                config.get('file_path'),
                tokenizer,
                config.get('system_prompt'),
                config.get('pasted_text')
            )
            
            self.update_progress(
                status='training',
                progress=30,
                message='Starting training...',
                total_epochs=config.get('epochs', 3)
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir='./models/fine_tuned',
                overwrite_output_dir=True,
                num_train_epochs=config.get('epochs', 3),
                per_device_train_batch_size=config.get('batch_size', 4),
                learning_rate=config.get('learning_rate', 5e-5),
                logging_steps=10,
                save_steps=500,
                save_total_limit=2,
                prediction_loss_only=True,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # Custom trainer class to track progress
            class ProgressTrainer(Trainer):
                def __init__(self, training_service, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.training_service = training_service
                    self.start_time = time.time()
                
                def log(self, logs):
                    super().log(logs)
                    if self.training_service.should_stop:
                        self.control.should_training_stop = True
                        return
                    
                    # Update progress
                    current_epoch = logs.get('epoch', 0)
                    total_epochs = self.args.num_train_epochs
                    progress = min(30 + (current_epoch / total_epochs) * 60, 90)
                    
                    elapsed_time = time.time() - self.start_time
                    if current_epoch > 0:
                        time_per_epoch = elapsed_time / current_epoch
                        remaining_epochs = total_epochs - current_epoch
                        estimated_remaining = time_per_epoch * remaining_epochs
                    else:
                        estimated_remaining = 0
                    
                    self.training_service.update_progress(
                        progress=progress,
                        current_epoch=current_epoch,
                        loss=logs.get('train_loss', 0.0),
                        estimated_time_remaining=estimated_remaining,
                        message=f'Training epoch {current_epoch:.1f}/{total_epochs}'
                    )
            
            # Create trainer
            trainer = ProgressTrainer(
                self,
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )
            
            # Start training
            logger.info("Starting training...")
            trainer.train()
            
            if not self.should_stop:
                self.update_progress(
                    status='saving',
                    progress=95,
                    message='Saving model...'
                )
                
                # Save the model
                model_save_path = './models/fine_tuned_final'
                os.makedirs(model_save_path, exist_ok=True)
                trainer.save_model(model_save_path)
                tokenizer.save_pretrained(model_save_path)
                
                self.update_progress(
                    status='completed',
                    progress=100,
                    message='Training completed successfully!',
                    estimated_time_remaining=0
                )
            else:
                self.update_progress(
                    status='stopped',
                    progress=0,
                    message='Training stopped by user'
                )
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            self.update_progress(
                status='error',
                progress=0,
                message=f'Training failed: {str(e)}'
            )
        
        finally:
            self.is_training_flag = False
    
    def stop_training(self):
        """Stop the current training process"""
        self.should_stop = True
        self.update_progress(
            status='stopping',
            message='Stopping training...'
        )

