# Slow Go Mofo: AI Fine-Tuning Web App

A modern open-source app for fine-tuning LLMs (GPT-2-family) on your own data, with a powerful web UI and production-ready backend.

---

## Features

- **Upload Dataset**: Accepts `.txt`, `.csv`, `.jsonl`, or `.json` files for training.
- **System Prompt Only**: Train with just a single system prompt (no file needed).
- **Paste Text**: Train on pasted raw text (no file needed).
- **Model Selection**: Choose from several Hugging Face models (GPT-2, DialoGPT, etc).
- **CPU/GPU Toggle**: Run on either CPU or GPU (if available).
- **Estimation**: Get a time/sample count estimate before starting training.
- **Fine-tuning**: Real-time progress bar, live logs, start/stop, and error handling.
- **Chat**: Interact with your freshly-trained model in the browser.
- **Download Model**: Download the trained model as a ready-to-use zip.
- **Push to Hugging Face Hub**: Upload your model directly with one click.
- **Modern UX**: React (Vite + Tailwind), responsive UI, minimal setup.
- **Backend**: Flask, PyTorch, Transformers, SQLite, Docker-ready.

---

## Tech Stack

- **Backend**: Python 3.11+, Flask, Flask-SQLAlchemy, Flask-CORS, PyTorch, Hugging Face Transformers & Datasets, HuggingFace Hub, SQLite.
- **Frontend**: React 18, Vite, Tailwind CSS, lucide-react, clsx.
- **Packaging/Deployment**: Docker (multi-stage), Gunicorn.
- **Database**: SQLite (optional, for users/auth/future extensibility).

---

## Getting Started (Local Development)

### 1. Clone & Set Up Python Backend

```bash
git clone <repo-url>
cd <project-root>
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.sample .env  # Edit as needed.
```

#### Environment variables (edit `.env`):

- `SECRET_KEY`: Flask secret.
- `SQLALCHEMY_DATABASE_URI`: e.g. `sqlite:///app.db`
- `HF_TOKEN`: [Hugging Face access token](https://huggingface.co/settings/tokens) (needed for model push).
- `UPLOAD_FOLDER`: Where uploaded files are stored (default: `uploads`)
- `PORT`: Backend port (default: 5000)

### 2. Run Backend (Flask Dev Mode)

```bash
export FLASK_ENV=development
python main.py  # Runs on :5000
```

---

### 3. Frontend (Vite Dev Server)

```bash
cd client
npm install
npm run dev
```

Visit [http://localhost:5173](http://localhost:5173) — the frontend will auto-proxy `/api` requests to Flask on :5000.

---

## Production Build

1. **Build Frontend**:  
   ```bash
   cd client
   npm run build
   ```
   This outputs static assets to `static/` at the repo root.

2. **Run Backend with Gunicorn**:  
   ```bash
   gunicorn main:create_app -b 0.0.0.0:5000 --workers 2
   ```

---

## Docker

A multi-stage Dockerfile is coming soon. It will:

- Build the React frontend
- Copy static assets to the Python image
- Install all Python deps
- Run Gunicorn

To use (once added):

```bash
docker build -t slow-go-mofo .
docker run --env-file .env -p 5000:5000 slow-go-mofo
```

---

## Hugging Face Model Push

1. [Create a Hugging Face token](https://huggingface.co/settings/tokens) and set `HF_TOKEN` in your `.env`.
2. After training, go to the "Model" tab, enter the repo name (e.g. `username/my-finetuned-model`), and click **Push to HF**.
3. The app will create the repo (if needed) and upload all model artefacts.

---

## API Reference

All endpoints are under `/api/training/`:

| Endpoint              | Method | Description                                                         |
|-----------------------|--------|---------------------------------------------------------------------|
| /upload               | POST   | Upload a dataset file                                               |
| /estimate             | POST   | Estimate time/resources for training                                |
| /start                | POST   | Start training                                                      |
| /progress             | GET    | Get training progress/status                                       |
| /stop                 | POST   | Stop training                                                      |
| /models               | GET    | List available base models                                         |
| /chat                 | POST   | Generate a reply with the fine-tuned model                         |
| /download             | GET    | Download the trained model as a zip                                |
| /push_hf              | POST   | Push trained model to Hugging Face Hub                             |

---

## Folder Structure

```
.
├── main.py                # Flask entrypoint
├── src/                   # Backend Python package
│   ├── models/
│   ├── routes/
│   ├── services/
│   └── config.py
├── client/                # React (Vite + Tailwind) frontend
│   ├── src/
│   ├── index.html
│   └── ...
├── static/                # Built frontend (after npm run build)
├── uploads/               # Uploaded files (git-ignored)
├── requirements.txt
├── .env.sample
└── README.md
```

---

## License

[SPDX-License-Identifier: MIT]  
*Add your details here.*

---