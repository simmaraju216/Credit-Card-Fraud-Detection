import pandas as pd
import uuid
import base64
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, flash

# ================= ML =================
from sklearn.ensemble import IsolationForest
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ================= GRAPHS =================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= APP =================
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index.html')

# ================= DASHBOARD =================
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', username='Guest')

@app.route('/home')
def home():
    return redirect(url_for('dashboard'))


@app.route('/profile')
def profile():
    return render_template('profile.html', username='Guest', user_data={})


@app.route('/profile/update', methods=['POST'])
def update_profile():
    flash("Profile updates are disabled in this version")
    return redirect(url_for('profile'))


@app.route('/admin')
def admin_page():
    return render_template(
        'admin.html',
        username='Guest',
        fraudulent_count=0,
        non_fraudulent_count=0,
        total_transactions=0,
        iso_forest_accuracy=0,
        iso_forest_error=0,
        iso_forest_classification_report='Upload a file to generate results.',
        svm_accuracy=0,
        svm_error=0,
        svm_classification_report='Upload a file to generate results.',
        logistic_accuracy=0,
        logistic_error=0,
        logistic_classification_report='Upload a file to generate results.',
        accuracy_img='img4.jpg',
        error_img='img4.jpg'
    )

# ================= ML PREDICTION =================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash("No file uploaded")
        return redirect(url_for('admin_page'))

    file = request.files['file']
    if file.filename == '':
        flash("No selected file")
        return redirect(url_for('admin_page'))

    data = pd.read_csv(file)

    if 'Class' in data.columns:
        target_col = 'Class'
    elif 'Approved' in data.columns:
        target_col = 'Approved'
    else:
        flash("Dataset must contain 'Class' or 'Approved'")
        return redirect(url_for('admin_page'))

    fraudulent_count = int((data[target_col] == 1).sum())
    non_fraudulent_count = int((data[target_col] == 0).sum())
    total_transactions = fraudulent_count + non_fraudulent_count

    if len(data) > 50000:
        data = data.sample(n=50000, random_state=42)

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Keep the existing split behavior for normal datasets, but fall back to an unstratified
    # split when the uploaded CSV is too small or too imbalanced for stratification.
    split_stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=split_stratify, random_state=42
    )

    iso_model = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
    iso_model.fit(X_train)
    iso_preds = [1 if p == -1 else 0 for p in iso_model.predict(X_test)]
    iso_accuracy = accuracy_score(y_test, iso_preds)
    iso_error = 1 - iso_accuracy
    iso_report = classification_report(y_test, iso_preds)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_model = LinearSVC(class_weight='balanced', max_iter=5000)
    svm_model.fit(X_train_scaled, y_train)
    svm_preds = svm_model.predict(X_test_scaled)

    svm_accuracy = accuracy_score(y_test, svm_preds)
    svm_error = 1 - svm_accuracy
    svm_report = classification_report(y_test, svm_preds)

    lr_model = LogisticRegression(class_weight='balanced', max_iter=300, n_jobs=-1)
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)

    lr_accuracy = accuracy_score(y_test, lr_preds)
    lr_error = 1 - lr_accuracy
    lr_report = classification_report(y_test, lr_preds)

    # ================= BAR GRAPHS =================
    models = ['Isolation Forest', 'SVM', 'Logistic Regression']
    accuracies = [iso_accuracy, svm_accuracy, lr_accuracy]
    errors = [iso_error, svm_error, lr_error]

    # Render the graphs in memory so deployment platforms with read-only filesystems do not fail.
    accuracy_buffer = BytesIO()
    plt.figure(figsize=(8, 5))
    plt.bar(models, accuracies, color=['#2563eb', '#16a34a', '#f59e0b'])
    plt.ylim(0, 1)
    plt.savefig(accuracy_buffer, format='png')
    plt.close()
    accuracy_buffer.seek(0)
    accuracy_img = f"data:image/png;base64,{base64.b64encode(accuracy_buffer.getvalue()).decode('utf-8')}"

    error_buffer = BytesIO()
    plt.figure(figsize=(8, 5))
    plt.bar(models, errors, color=['#dc2626', '#7c3aed', '#0d9488'])
    plt.ylim(0, 1)
    plt.savefig(error_buffer, format='png')
    plt.close()
    error_buffer.seek(0)
    error_img = f"data:image/png;base64,{base64.b64encode(error_buffer.getvalue()).decode('utf-8')}"

    return render_template(
        'admin.html',
        username='Guest',
        fraudulent_count=fraudulent_count,
        non_fraudulent_count=non_fraudulent_count,
        total_transactions=total_transactions,

        iso_forest_accuracy=iso_accuracy,
        iso_forest_error=iso_error,
        iso_forest_classification_report=iso_report,

        svm_accuracy=svm_accuracy,
        svm_error=svm_error,
        svm_classification_report=svm_report,

        logistic_accuracy=lr_accuracy,
        logistic_error=lr_error,
        logistic_classification_report=lr_report,

        accuracy_img=accuracy_img,
        error_img=error_img
    )

if __name__ == '__main__':
    app.run(debug=True)

#venv\Scripts\python app.py