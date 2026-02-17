from flask import Flask, render_template, request, session, redirect, url_for, make_response, jsonify, abort
import pickle
import os
from functools import wraps
import re
from datetime import datetime, timedelta
import json
import logging
from logging.handlers import RotatingFileHandler
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(BASE_DIR, "frontend", "templates")
app = Flask(__name__, template_folder=TEMPLATES_DIR)
# Use an environment variable for production secrets; fall back to a random key for local dev
app.secret_key = os.environ.get('FLASK_SECRET') or os.urandom(24).hex()

# Logging setup
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, 'app.log')
handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(filename)s:%(lineno)d]'))
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# In-memory storage
scan_history = {}
whitelist = set()
blacklist = set()

# API keys persisted to disk
API_KEYS_PATH = os.path.join(BASE_DIR, 'api_keys.json')
api_keys = {}  # maps api_key -> username

def load_api_keys():
    global api_keys
    try:
        if os.path.exists(API_KEYS_PATH):
            with open(API_KEYS_PATH, 'r') as f:
                api_keys = json.load(f)
        else:
            api_keys = {}
    except Exception as e:
        app.logger.error("Failed to load API keys: %s", e)
        api_keys = {}

def save_api_keys():
    try:
        with open(API_KEYS_PATH, 'w') as f:
            json.dump(api_keys, f)
    except Exception as e:
        app.logger.error("Failed to save API keys: %s", e)

# Model globals
model = None
vectorizer = None
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# Demo credentials (in production, use a proper database). Passwords are hashed for basic security.
VALID_USERS = {
    'admin': generate_password_hash('admin123'),
    'user': generate_password_hash('user123'),
    'test': generate_password_hash('test123')
}

# Simple in-memory rate limiter: key -> (count, reset_time)
rate_limits = {}

def rate_limit(max_calls=10, window_seconds=60):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            key = session.get('user') or request.headers.get('X-API-Key') or request.remote_addr
            now = datetime.now()
            count, reset = rate_limits.get(key, (0, now + timedelta(seconds=window_seconds)))
            if now > reset:
                count, reset = (0, now + timedelta(seconds=window_seconds))
            if count >= max_calls:
                app.logger.warning("Rate limit exceeded for key=%s", key)
                return jsonify({"error": "Rate limit exceeded"}), 429
            rate_limits[key] = (count + 1, reset)
            return f(*args, **kwargs)
        return wrapped
    return decorator


def load_model():
    """Attempt to load model and vectorizer from disk into globals."""
    global model, vectorizer
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
            with open(MODEL_PATH, "rb") as mf:
                model = pickle.load(mf)
            with open(VECT_PATH, "rb") as vf:
                vectorizer = pickle.load(vf)
        else:
            model = None
            vectorizer = None
    except Exception:
        model = None
        vectorizer = None


# Try initial load (useful when starting the server)
load_model()
# Load persisted API keys (if present)
load_api_keys()


def login_required(f):
    """Decorator to check if user is logged in."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route("/")
def index():
    """Redirect to login if not authenticated, else to home."""
    if 'user' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('home'))


@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle login page."""
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        
        if username in VALID_USERS and check_password_hash(VALID_USERS[username], password):
            session['user'] = username
            return redirect(url_for('home'))
        else:
            error = "Invalid username or password"
    
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    """Handle logout."""
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route("/home")
@login_required
def home():
    missing = model is None or vectorizer is None
    import time
    version = int(time.time())
    response = make_response(render_template("index.html", model_missing=missing, version=version))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/test")
def test():
    """Test page to show all new features"""
    return render_template("test.html")


@app.route('/ping')
def ping():
    """Lightweight endpoint to verify server is up"""
    return 'ok', 200


@app.route("/history")
@login_required
def history():
    user = session.get('user')
    user_history = scan_history.get(user, [])
    return jsonify({"history": user_history})


@app.route("/export")
@login_required
def export():
    user = session.get('user')
    user_history = scan_history.get(user, [])
    csv_data = "Timestamp,Threat,Confidence,Sender,Email Preview\n"
    for scan in user_history:
        csv_data += f"{scan['timestamp']},{scan['threat']},{scan['confidence']},{scan['sender']},{scan['email']}\n"
    
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment;filename=scan_history.csv"
    response.headers["Content-Type"] = "text/csv"
    return response


@app.route("/whitelist", methods=["POST"])
@login_required
def add_whitelist():
    email = request.json.get("email", "").lower().strip()
    whitelist.add(email)
    return jsonify({"status": "added", "email": email})


@app.route("/blacklist", methods=["POST"])
@login_required
def add_blacklist():
    email = request.json.get("email", "").lower().strip()
    blacklist.add(email)
    return jsonify({"status": "added", "email": email})


@app.route("/stats")
@login_required
def stats():
    user = session.get('user')
    user_history = scan_history.get(user, [])
    spam_count = sum(1 for s in user_history if s['threat'] == 'SPAM')
    safe_count = sum(1 for s in user_history if s['threat'] == 'SAFE')
    avg_confidence = sum(float(s['confidence']) for s in user_history) / len(user_history) if user_history else 0
    
    # Get threat distribution
    threat_types = {}
    for scan in user_history:
        threat = scan.get('threat', 'Unknown')
        threat_types[threat] = threat_types.get(threat, 0) + 1
    
    top_threat = max(threat_types, key=threat_types.get) if threat_types else "None"
    last_scan = user_history[-1]['timestamp'] if user_history else "Never"
    
    return jsonify({
        "total_scans": len(user_history),
        "spam_detected": spam_count,
        "safe_emails": safe_count,
        "avg_confidence": f"{avg_confidence:.1f}",
        "threat_percentage": f"{(spam_count / len(user_history) * 100):.1f}" if user_history else "0",
        "top_threat": top_threat,
        "last_scan": last_scan
    })


@app.route("/api/generate-key", methods=["POST"])
@login_required
def generate_api_key():
    import secrets
    api_key = secrets.token_hex(32)
    user = session.get('user')
    api_keys[api_key] = user
    save_api_keys()
    app.logger.info("API key generated for user %s", user)
    return jsonify({"api_key": api_key, "status": "generated"})


@app.route("/api/threat-chart")
@login_required
def threat_chart():
    user = session.get('user')
    user_history = scan_history.get(user, [])
    
    phishing_total = sum(float(s.get('phishing_score', 0)) for s in user_history)
    malware_total = sum(float(s.get('malware_score', 0)) for s in user_history)
    spam_total = sum(1 for s in user_history if s['threat'] == 'SPAM') * 20
    
    return jsonify({
        "phishing": phishing_total // len(user_history) if user_history else 0,
        "malware": malware_total // len(user_history) if user_history else 0,
        "spam": spam_total // len(user_history) if user_history else 0
    })


@app.route("/api/quick-templates")
def quick_templates():
    templates = [
        {"name": "Safe Email", "text": "Hello,\n\nJust checking in. How are you doing today?\n\nBest regards,\nJohn Smith\njohn@company.com"},
        {"name": "Phishing Test", "text": "URGENT: Verify your account!\n\nClick here to confirm your identity: http://suspicious-link.xyz\n\nDon't delay!"},
        {"name": "Malware Test", "text": "Check out this file: document.exe\n\nPlease download and execute immediately!"},
    ]
    return jsonify(templates)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if model is None or vectorizer is None:
        load_model()

    if model is None or vectorizer is None:
        message = "Model not found. Run training script."
        return render_template("index.html", prediction=message)

    email_text = request.form.get("email", "").strip()
    if not email_text:
        return render_template("index.html", prediction="❌ Please enter email content")
    
    data = vectorizer.transform([email_text])
    prediction = model.predict(data)
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(data)[0]
        confidence = max(proba) * 100
    else:
        confidence = 95
    
    # Extract links from email
    links = re.findall(r'https?://[^\s]+', email_text)
    
    # Extract attachments (look for file extensions)
    attachments = re.findall(r'[\w\-\.]+\.(exe|zip|rar|scr|bat|com|pif|vbs|docm|xlsm|pdf|jpg|png)', email_text, re.IGNORECASE)
    dangerous_files = [f for f in attachments if f.lower() in ['exe', 'zip', 'rar', 'scr', 'bat', 'com', 'pif', 'vbs', 'docm', 'xlsm']]
    
    # Analyze threat components
    phishing_indicators = len(re.findall(r'verify.*account|confirm.*identity|update.*password|click.*here|urgent.*action', email_text, re.IGNORECASE))
    malware_indicators = len(re.findall(r'\.exe|\.zip|\.scr|download|attachment|execute', email_text, re.IGNORECASE))
    
    # Calculate component scores
    phishing_score = min(phishing_indicators * 15, 100)
    malware_score = min((len(dangerous_files) * 20 + malware_indicators * 10), 100)
    
    # Extract sender info
    sender_match = re.search(r'from:?\s*([^\n<]+)', email_text, re.IGNORECASE)
    sender = sender_match.group(1).strip() if sender_match else "Unknown"
    
    # Check whitelist/blacklist
    if sender.lower() in whitelist:
        result = f"✅ WHITELISTED ({sender})"
        threat = "SAFE"
    elif sender.lower() in blacklist:
        result = f"⚠️ BLACKLISTED ({sender})"
        threat = "SPAM"
    elif prediction[0] == 1:
        result = f"⚠️ THREAT DETECTED ({confidence:.1f}% confidence)"
        threat = "SPAM"
    else:
        result = f"✅ SAFE EMAIL ({confidence:.1f}% confidence)"
        threat = "SAFE"
    
    # Store in history
    user = session.get('user', 'Unknown')
    if user not in scan_history:
        scan_history[user] = []
    
    scan_history[user].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "threat": threat,
        "confidence": f"{confidence:.1f}",
        "email": email_text[:80],
        "sender": sender,
        "links": links,
        "attachments": attachments,
        "phishing_score": phishing_score,
        "malware_score": malware_score
    })
    
    return render_template("index.html", prediction=result, links=links, sender=sender, threat=threat, 
                         attachments=dangerous_files, phishing_score=phishing_score, malware_score=malware_score, confidence=confidence)


@app.route("/api/domain-reputation", methods=["POST"])
@login_required
def domain_reputation():
    """Check domain reputation (mock implementation)"""
    domain = request.json.get("domain", "").strip().lower()
    
    if not domain:
        return jsonify({"error": "Domain required"}), 400
    
    # Mock reputation check
    suspicious_domains = ['bit.ly', 'tinyurl.com', 'goo.gl', 'short.link']
    is_suspicious = any(s in domain for s in suspicious_domains)
    
    return jsonify({
        "domain": domain,
        "reputation": "suspicious" if is_suspicious else "safe",
        "score": 35 if is_suspicious else 95,
        "blacklisted": is_suspicious,
        "last_checked": datetime.now().isoformat()
    })


@app.route("/api/analyze-headers", methods=["POST"])
@login_required
def analyze_headers():
    """Analyze email headers for spoofing/authentication"""
    headers = request.json.get("headers", "").strip()
    
    if not headers:
        return jsonify({"error": "Headers required"}), 400
    
    # Check for authentication indicators
    has_spf = 'spf=' in headers.lower()
    has_dkim = 'dkim=' in headers.lower()
    has_dmarc = 'dmarc=' in headers.lower()
    
    # Check for suspicious patterns
    spoofing_indicators = len(re.findall(r'received.*spoofed|forged|unknown', headers, re.IGNORECASE))
    
    return jsonify({
        "spf": "Pass" if has_spf else "Missing",
        "dkim": "Pass" if has_dkim else "Missing",
        "dmarc": "Pass" if has_dmarc else "Missing",
        "spoofing_risk": "High" if spoofing_indicators > 0 else "Low",
        "authenticity_score": 85,
        "warnings": ["Missing DKIM"] if not has_dkim else []
    })


@app.route("/api/attachment-scan", methods=["POST"])
@login_required
def scan_attachment():
    """Scan attachment for threats (mock implementation)"""
    filename = request.json.get("filename", "").strip()
    
    if not filename:
        return jsonify({"error": "Filename required"}), 400
    
    # Known dangerous extensions
    dangerous_exts = ['exe', 'zip', 'rar', 'scr', 'bat', 'cmd', 'com', 'pif', 'vbs', 'js', 'jar']
    ext = filename.split('.')[-1].lower() if '.' in filename else ''
    
    is_dangerous = ext in dangerous_exts
    
    return jsonify({
        "filename": filename,
        "safe": not is_dangerous,
        "risk_level": "Critical" if is_dangerous else "Safe",
        "extension": ext,
        "threat_type": "Executable" if is_dangerous else "None"
    })


@app.route("/admin/reload-model", methods=["POST"])
@login_required
def reload_model_endpoint():
    """Reload the ML model and vectorizer"""
    global model, vectorizer
    load_model()
    
    return jsonify({
        "status": "Model reloaded" if model is not None else "Failed to load model",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/advanced-analytics")
@login_required
def advanced_analytics():
    """Get advanced analytics for dashboard"""
    user = session.get('user')
    user_history = scan_history.get(user, [])
    
    if not user_history:
        return jsonify({
            "top_threat": "N/A",
            "last_scan": "Never",
            "threat_types": {},
            "temporal_data": [],
            "most_common_sender": "N/A"
        })
    
    # Analyze threat distribution
    threat_types = {}
    for scan in user_history:
        threat = scan.get('threat', 'Unknown')
        threat_types[threat] = threat_types.get(threat, 0) + 1
    
    # Get most common threat type
    top_threat = max(threat_types, key=threat_types.get) if threat_types else "N/A"
    
    # Get most common sender
    senders = [s.get('sender', 'Unknown') for s in user_history]
    sender_counts = {}
    for sender in senders:
        sender_counts[sender] = sender_counts.get(sender, 0) + 1
    most_common = max(sender_counts, key=sender_counts.get) if sender_counts else "N/A"
    
    return jsonify({
        "top_threat": top_threat,
        "last_scan": user_history[-1]['timestamp'] if user_history else "Never",
        "threat_types": threat_types,
        "total_threats": threat_types.get('SPAM', 0),
        "most_common_sender": most_common
    })


@app.route("/api/predict", methods=["POST"])
@rate_limit(max_calls=20, window_seconds=60)
def api_predict():
    """API endpoint for email prediction"""
    if model is None or vectorizer is None:
        load_model()
    
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not available"}), 503
    
    data = request.json
    email_text = data.get("email", "").strip()
    api_key = data.get("api_key") or request.headers.get("X-API-Key")
    
    if not email_text:
        return jsonify({"error": "Email content required"}), 400
    
    # Validate API key if provided
    if api_key and api_key not in api_keys:
        return jsonify({"error": "Invalid API key"}), 401
    
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(email_vector)[0]
        confidence = max(proba) * 100
    else:
        confidence = 90
    
    is_spam = prediction[0] == 1
    
    return jsonify({
        "spam": is_spam,
        "confidence": f"{confidence:.1f}",
        "result": "SPAM" if is_spam else "SAFE",
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    app.run(debug=True)