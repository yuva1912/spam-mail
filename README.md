# Spam Mail Detection (backend)

Setup and run (Windows PowerShell):

1. Install Python from https://www.python.org/ and enable "Add Python to PATH".

2. From project root run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

3. Train the model (creates `spam_model.pkl` and `vectorizer.pkl`):

```powershell
python backend/train_model.py
```

4. Run the Flask server:

```powershell
python backend/app.py
```

Notes:
- If the model files are missing, the app still starts and will show a message prompting you to run the training script.
- Template is at `backend/frontend/templates/index.html`.
