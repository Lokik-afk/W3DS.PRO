import json
import os
import requests
import hashlib
from flask import render_template, redirect, request, send_file, session, flash, url_for, jsonify
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from datetime import datetime, timedelta
from app import app
from timeit import default_timer as timer
from hashlib import sha256
import urllib.parse
from web3 import Web3
from solcx import compile_source, install_solc, set_solc_version
from dotenv import load_dotenv
from flask import session, redirect, url_for

install_solc('0.8.0')
set_solc_version('0.8.0')

load_dotenv()

# MongoDB Atlas Configuration
username = os.getenv("MONGO_USER")
password = os.getenv("MONGO_PASS")
cluster_url = os.getenv("MONGO_CLUSTER")
encoded_password = urllib.parse.quote_plus(password)

MONGO_URI = f"mongodb+srv://{username}:{encoded_password}@{cluster_url}/vault_db?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client.vault_db
users = db.users

# Sepolia Testnet Configuration
SEPOLIA_RPC = os.getenv("SEPOLIA_RPC", "https://rpc.sepolia.org")
WEB3_PROVIDER = Web3.HTTPProvider(SEPOLIA_RPC)
w3 = Web3(WEB3_PROVIDER)

# Deployer account from environment
DEPLOYER_PRIVATE_KEY = os.getenv("DEPLOYER_PRIVATE_KEY")
DEPLOYER_ADDRESS = os.getenv("DEPLOYER_ADDRESS")

# Compile contract
def compile_contract(source_path):
    with open(source_path, 'r') as f:
        source_code = f.read()

    compiled_sol = compile_source(
        source_code,
        output_values=['abi', 'bin'],
        solc_version='0.8.0'
    )

    contract_id, contract_interface = compiled_sol.popitem()

    return {
        'abi': contract_interface['abi'],
        'bytecode': contract_interface['bin']
    }

# Deploy contract for each user
def deploy_vault_contract(username):
    compiled_contract = compile_contract('VaultStorage.sol')

    contract = w3.eth.contract(
        abi=compiled_contract['abi'],
        bytecode=compiled_contract['bytecode']
    )

    deployer_address = DEPLOYER_ADDRESS
    w3.eth.default_account = deployer_address

    txn = contract.constructor(username).build_transaction({
        'from': deployer_address,
        'nonce': w3.eth.get_transaction_count(deployer_address),
        'gas': 3000000,
        'gasPrice': w3.to_wei('10', 'gwei')
    })

    signed_txn = w3.eth.account.sign_transaction(txn, private_key=DEPLOYER_PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    return receipt.contractAddress

# File storage configuration
request_tx = []
files = {}
UPLOAD_FOLDER = "app/static/Uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ADDR = "http://127.0.0.1:8800"
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(days=7)

# Helper function for SHA-256 passkey
def generate_passkey(password, salt=None):
    if not salt:
        salt = os.urandom(16).hex()
    passkey = sha256((password + salt).encode()).hexdigest()
    return passkey, salt

# Get blockchain transactions
def get_tx_req():
    global request_tx
    chain_addr = f"{ADDR}/chain"
    resp = requests.get(chain_addr)
    if resp.status_code == 200:
        content = []
        chain = json.loads(resp.content.decode())
        for block in chain["chain"]:
            for trans in block["transactions"]:
                trans["index"] = block["index"]
                trans["hash"] = block["prev_hash"]
                content.append(trans)
        request_tx = sorted(content, key=lambda k: k["hash"], reverse=True)

# Routes
@app.route("/")
def dash():
    return render_template("dash.html")

@app.route("/dashboard")
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    get_tx_req()
    return render_template("index.html",
                         title="FileStorage",
                         subtitle="A Decentralized Network for File Storage/Sharing",
                         node_address=ADDR,
                         request_tx=request_tx)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    try:
        username = request.form.get("username").strip()
        salt = request.form.get("salt").strip()
        passkey = request.form.get("passkey").strip()

        if not all([username, salt, passkey]):
            return jsonify({"status": "error", "message": "All fields are required"}), 400

        user = users.find_one({"username": username})
        if not user:
            return jsonify({"status": "error", "message": "Invalid credentials"}), 401

        if salt != user["salt"] or passkey != user["passkey"]:
            return jsonify({"status": "error", "message": "Invalid credentials"}), 401

        session["username"] = username
        return jsonify({"status": "success", "redirect": "/dashboard"})

    except Exception as e:
        app.logger.error(f"Login error: {str(e)}")
        return jsonify({"status": "error", "message": "An error occurred"}), 500

@app.route("/register", methods=["POST"])
def register():
    try:
        username = request.form.get("username").strip()
        email = request.form.get("email").strip()
        password = request.form.get("password").strip()

        if not all([username, email, password]):
            return jsonify({"status": "error", "message": "All fields are required"}), 400

        if users.find_one({"$or": [{"username": username}, {"email": email}]}):
            return jsonify({"status": "error", "message": "Username or email already exists"}), 400

        passkey, salt = generate_passkey(password)
        contract_address = deploy_vault_contract(username)

        users.insert_one({
            "username": username,
            "email": email,
            "passkey": passkey,
            "salt": salt,
            "contract_address": contract_address,
            "created_at": datetime.utcnow()
        })

        return jsonify({
            "status": "success",
            "message": "User registered successfully",
            "salt": salt,
            "passkey": passkey,
            "contract_address": contract_address
        })

    except Exception as e:
        app.logger.error(f"Registration error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Registration failed: {str(e)}"
        }), 500


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/submit", methods=["POST"])
def submit():
    if 'username' not in session:
        return redirect(url_for('login'))

    try:
        start = timer()
        user = session['username']
        up_file = request.files["v_file"]

        filename = secure_filename(up_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        up_file.save(filepath)

        files[filename] = os.path.join(app.root_path, "static", "Uploads", filename)
        file_size = os.path.getsize(filepath)

        post_object = {
            "user": user,
            "v_file": filename,
            "file_data": str(up_file.stream.read()),
            "file_size": file_size
        }

        requests.post(f"{ADDR}/new_transaction", json=post_object)

        end = timer()
        app.logger.info(f"File upload took {end - start} seconds")
        return redirect("/dashboard")

    except Exception as e:
        app.logger.error(f"File upload error: {str(e)}")
        flash("File upload failed")
        return redirect("/dashboard")

@app.route("/submit/<string:filename>", methods=["GET"])
def download_file(filename):
    if 'username' not in session:
        return redirect(url_for('login'))

    if filename not in files:
        flash("File not found")
        return redirect("/dashboard")

    return send_file(files[filename], as_attachment=True)
