import os
import json
import time
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline
)
from datasets import Dataset
import pandas as pd
from threading import Lock
import logging
import shutil
import tempfile
from huggingface_hub import HfApi, upload_folder, create_repo
from tqdm import tqdm

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

        self.model_save_path = './models/fine_tuned_final'
        self.inference_model = None
        self.inference_tokenizer = None

    def is_training(self):
        return self.is_training_flag

    def get_progress(self):
        with self.training_lock:
            return self.progress_data.copy()

    def update_progress(self, **kwargs):
        with self.training_lock:
            self.progress_data.update(kwargs)

    def estimate_training_time(self, file_path, device, model_name):
        try:
            file_size = os.path.getsize(file_path)
            sample_count = self._count_samples(file_path)
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
            factor = base_factors.get(device, base_factors['cpu']).get(model_name, 2.0)
            epochs = 3
            estimated_seconds = sample_count * factor * epochs
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
        try:
            if file_path and file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    return len([line for line in lines if line.strip()])
            elif file_path and file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                return len(df)
            elif file_path and file_path.endswith(('.jsonl', '.json')):
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
            return 100
        except Exception as e:
            logger.error(f"Error counting samples: {e}")
            return 100

    def _prepare_dataset(self, file_path=None, tokenizer=None, system_prompt=None, pasted_text=None):
        try:
            texts = []
            if pasted_text:
                texts = [pasted_text]
            elif system_prompt:
                texts = [system_prompt]
            elif file_path and file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f.readlines() if line.strip()]
            elif file_path and file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                text_column = df.select_dtypes(include=['object']).columns[0]
                texts = df[text_column].dropna().tolist()
            elif file_path and file_path.endswith(('.jsonl', '.json')):
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
            if not texts:
                raise ValueError("No training data supplied")
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
            dataset = Dataset.from_list(tokenized_texts)
            return dataset
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise e

    def start_training(self, config):
        try:
            self.is_training_flag = True
            self.should_stop = False
            self.update_progress(
                status='initializing',
                progress=0,
                message='Initializing training...',
                start_time=time.time()
            )
            model_name = config.get('model_name')
            device = config.get('device')
            file_path = config.get('file_path')
            system_prompt = config.get('system_prompt')
            pasted_text = config.get('pasted_text')
            if not (file_path or system_prompt or pasted_text):
                raise ValueError("Must provide at least one of: file_path, system_prompt, or pasted_text for training.")

            logger.info(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
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
            dataset = self._prepare_dataset(
                file_path=file_path,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                pasted_text=pasted_text
            )
            self.update_progress(
                status='training',
                progress=30,
                message='Starting training...',
                total_epochs=config.get('epochs', 3)
            )
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
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
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
            trainer = ProgressTrainer(
                self,
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )
            logger.info("Starting training...")
            trainer.train()
            if not self.should_stop:
                self.update_progress(
                    status='saving',
                    progress=95,
                    message='Saving model...'
                )
                model_save_path = self.model_save_path
                os.makedirs(model_save_path, exist_ok=True)
                trainer.save_model(model_save_path)
                tokenizer.save_pretrained(model_save_path)
                self.inference_model = None  # Invalidate cache
                self.inference_tokenizer = None
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
        self.should_stop = True
        self.update_progress(
            status='stopping',
            message='Stopping training...'
        )

    # --- Inference & Model Management Extensions ---

    def get_latest_model_path(self) -> str:
        """Return model save path if exists, else raise FileNotFoundError."""
        if os.path.isdir(self.model_save_path) and os.path.exists(os.path.join(self.model_save_path, "config.json")):
            return self.model_save_path
        raise FileNotFoundError("No fine-tuned model artefacts found. Please train a model first.")

    def load_for_inference(self):
        """Load the model/tokenizer for inference if not already loaded."""
        if self.inference_model is not None and self.inference_tokenizer is not None:
            return self.inference_model, self.inference_tokenizer
        model_path = self.get_latest_model_path()
        self.inference_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.inference_model = AutoModelForCausalLM.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inference_model = self.inference_model.to(device)
        return self.inference_model, self.inference_tokenizer

    def generate_response(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate a model reply to a prompt with the fine-tuned model."""
        model, tokenizer = self.load_for_inference()
        device = 0 if torch.cuda.is_available() else -1
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        outputs = generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        return outputs[0]["generated_text"]

    def create_zip(self) -> str:
        """Create a zip file of the model artefacts and return its path."""
        model_dir = self.get_latest_model_path()
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            zip_path = tmp.name
        shutil.make_archive(zip_path.replace(".zip", ""), "zip", model_dir)
        # make_archive adds .zip again
        final_path = zip_path.replace(".zip", "") + ".zip"
        return final_path

    def push_to_hf(self, repo_name: str, private: bool, token: str) -> None:
        """Push the latest model to Hugging Face Hub."""
        model_dir = self.get_latest_model_path()
        api = HfApi(token=token)
        # Create repo (won't error if already exists for user)
        create_repo(repo_id=repo_name, token=token, private=private, exist_ok=True)
        upload_folder(
            repo_id=repo_name,
            folder_path=model_dir,
            allow_patterns=["*.bin", "*.json", "*.txt", "*.model", "*.py"],
            token=token,
            repo_type="model",
        )