from flask import Flask, render_template, request, session, redirect, url_for, flash, send_file
import pickle
import numpy as np
import sqlite3
import csv
import io

app = Flask(__name__)
app.secret_key = "supersecretkey"  # only once, keep it at the top

import pdfkit
path_wkhtmltopdf = "/usr/bin/wkhtmltopdf"
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
import os
import matplotlib
matplotlib.use('Agg')  # headless backend for PDF generation
import matplotlib.pyplot as plt

def generate_charts(transactions):
    # Bar chart: Safe vs Fraud
    safe_count = sum(1 for t in transactions if t['result'] != "Fraud Detected")
    fraud_count = sum(1 for t in transactions if t['result'] == "Fraud Detected")
    
    # Transactions Bar Chart
    plt.figure(figsize=(6,4))
    plt.bar(['Safe', 'Fraud'], [safe_count, fraud_count], color=['#28a745', '#dc3545'])
    plt.title("Transactions Overview")
    plt.ylabel("Count")
    os.makedirs('static/images', exist_ok=True)
    transactions_chart = 'static/images/transactions_chart.png'
    plt.savefig(transactions_chart, bbox_inches='tight')
    plt.close()

    # Pie Chart
    plt.figure(figsize=(4,4))
    plt.pie([safe_count, fraud_count], labels=['Safe','Fraud'], autopct='%1.1f%%', colors=['#28a745','#dc3545'])
    plt.title("Fraud vs Safe")
    fraud_pie_chart = 'static/images/fraud_pie_chart.png'
    plt.savefig(fraud_pie_chart, bbox_inches='tight')
    plt.close()

    return transactions_chart, fraud_pie_chart


def get_db_connection():
    conn = sqlite3.connect('transactions.db')  # match your DB
    conn.row_factory = sqlite3.Row
    return conn



with open('model/fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

def init_db():
    conn = sqlite3.connect('transactions.db')
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT UNIQUE,
            amount REAL,
            transaction_type TEXT,
            location TEXT,
            device TEXT,
            result TEXT
        )
    """)

    # Add alerted column if it doesn't exist
    cursor.execute("PRAGMA table_info(transactions)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'alerted' not in columns:
        cursor.execute("ALTER TABLE transactions ADD COLUMN alerted INTEGER DEFAULT 0")

    conn.commit()
    conn.close()
    print("Database Initialized Successfully!")

init_db()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':

        transaction_id = request.form.get('transaction_id')
        amount = float(request.form.get('amount'))
        transaction_type = request.form.get('transaction_type')
        location = request.form.get('location')
        device = request.form.get('device')

        # Feature engineering
        is_online = 1 if transaction_type == "Online" else 0
        local_cities = ["new york", "los angeles", "chicago"]
        is_foreign = 0 if location.lower() in local_cities else 1
        is_high_risk_device = 1 if device == "Mobile" else 0

        features = np.array([[amount, is_online, is_foreign, is_high_risk_device]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        result = "Fraud Detected" if prediction == 1 else "Transaction Safe"
        risk_score = round(probability * 100, 2)

        try:
            conn = sqlite3.connect('transactions.db')
            cursor = conn.cursor()

            # Insert transaction
            cursor.execute("""
                INSERT INTO transactions
                (transaction_id, amount, transaction_type, location, device, result)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                transaction_id,
                amount,
                transaction_type,
                location,
                device,
                result
            ))

            # If fraud, mark as new alert (alerted = 0)
            if result == "Fraud Detected":
                cursor.execute("""
                    UPDATE transactions
                    SET alerted = 0
                    WHERE transaction_id = ?
                """, (transaction_id,))

            conn.commit()

        except sqlite3.IntegrityError:
            conn.close()
            return "Error: Transaction ID already exists!"

        finally:
            conn.close()

        return render_template(
            'result.html',
            data=request.form,
            result=result,
            risk_score=risk_score
        )

    return render_template('detect.html')

@app.route('/learn_more')
def learn_more():
    return render_template('learn_more.html')

@app.route('/transactions')
def all_transactions():
    conn = sqlite3.connect('transactions.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM transactions ORDER BY id DESC")
    transactions = cursor.fetchall()

    conn.close()

    return render_template('transactions.html', transactions=transactions)




@app.route('/transaction/<int:transaction_id>')
def transaction_detail(transaction_id):

    conn = sqlite3.connect('transactions.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM transactions WHERE id = ?", (transaction_id,))
    transaction = cursor.fetchone()

    conn.close()

    return render_template('transaction_detail.html', transaction=transaction)




@app.route('/locations')
def locations():
    conn = sqlite3.connect('transactions.db')
    cursor = conn.cursor()

    # Get total and fraud counts per location
    cursor.execute("""
        SELECT location, COUNT(*), SUM(CASE WHEN result='Fraud Detected' THEN 1 ELSE 0 END)
        FROM transactions
        GROUP BY location
    """)
    data = cursor.fetchall()
    conn.close()

    location_stats = []
    map_data = []

    # Coordinates of major Indian cities
    # Coordinates of major Indian cities + additional Maharashtra cities
    coords_dict = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Nagpur": (21.1458, 79.0882),
    "Indore": (22.7196, 75.8577),
    "Thane": (19.2183, 72.9781),
    "Bhopal": (23.2599, 77.4126),
    "Visakhapatnam": (17.6868, 83.2185),
    "Surat": (21.1702, 72.8311),
    "Vadodara": (22.3072, 73.1812),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    # Maharashtra additional cities
    "Satara": (17.6800, 73.9900),
    "Kolhapur": (16.7050, 74.2433),
    "Sangli": (16.8531, 74.5655),
    "Solapur": (17.6599, 75.9064),
    "Akluj": (17.9000, 75.0500)
    }

    for loc, total, fraud_cases in data:
        fraud_percent = round((fraud_cases / total) * 100, 2)  # properly indented
        location_stats.append([loc, total, fraud_cases, fraud_percent])

        # Normalize location name for coordinates
        loc_key = loc.title()
        if fraud_cases > 0 and loc_key in coords_dict:
            lat, lon = coords_dict[loc_key]
            map_data.append({
                "location": loc_key,
                "lat": lat,
                "lon": lon,
                "fraud_cases": fraud_cases,
                "risk_percent": fraud_percent
            })

    # Debug print
    print("Map data:", map_data)

    return render_template('locations.html',
                           location_stats=location_stats,
                           locations=[d['location'] for d in map_data],
                           counts=[d['fraud_cases'] for d in map_data],
                           map_data=map_data)

@app.route('/reports')
def reports():

    conn = sqlite3.connect('transactions.db')
    cursor = conn.cursor()

    # Total transactions
    cursor.execute("SELECT COUNT(*) FROM transactions")
    total_transactions = cursor.fetchone()[0]

    # Fraud count
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE result = 'Fraud Detected'")
    fraud_count = cursor.fetchone()[0]

    # Safe count
    safe_count = total_transactions - fraud_count

    fraud_percentage = round((fraud_count / total_transactions) * 100, 2) if total_transactions > 0 else 0

    conn.close()

    return render_template(
        'reports.html',
        total=total_transactions,
        fraud=fraud_count,
        safe=safe_count,
        fraud_percentage=fraud_percentage
    )

import csv
from flask import send_file
import io

@app.route('/download_report')
def download_report():
    conn = sqlite3.connect('transactions.db')
    cursor = conn.cursor()
    cursor.execute("SELECT transaction_id, amount, transaction_type, location, device, result FROM transactions")
    data = cursor.fetchall()
    conn.close()

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Transaction ID','Amount','Type','Location','Device','Result'])
    writer.writerows(data)
    output.seek(0)

    return send_file(io.BytesIO(output.getvalue().encode()),
                     mimetype='text/csv',
                     as_attachment=True,
                     download_name='fraud_report.csv')



@app.route('/download_report_pdf')
def download_report_pdf():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transactions ORDER BY id ASC")
    transactions = cursor.fetchall()
    
    total = len(transactions)
    fraud = sum(1 for t in transactions if t['result'] == "Fraud Detected")
    safe = total - fraud
    fraud_percentage = round((fraud / total) * 100, 2) if total > 0 else 0

    # Generate charts
    transactions_chart, fraud_pie_chart = generate_charts(transactions)

    # Render HTML
    rendered = render_template(
        'reports_pdf.html',
        total=total,
        fraud=fraud,
        safe=safe,
        fraud_percentage=fraud_percentage,
        transactions=transactions,
        transactions_chart=transactions_chart,
        fraud_pie_chart=fraud_pie_chart
    )

    # PDF generation
    options = {'enable-local-file-access': ''}
    pdf = pdfkit.from_string(rendered, False, configuration=config, options=options)

    return (pdf, 200, {
        'Content-Type': 'application/pdf',
        'Content-Disposition': 'attachment; filename="fraud_full_report.pdf"'
    })

@app.route('/alerts')
def alerts():
    # Connect to your database
    conn = get_db_connection()
    cursor = conn.cursor()

    # Query for fraudulent transactions
    query = "SELECT * FROM transactions WHERE result='Fraud Detected'"
    cursor.execute(query)
    alerts = cursor.fetchall()

    # Close the connection
    conn.close()

    # Render the alerts.html template and pass the alerts data
    return render_template('alerts.html', alerts=alerts)





@app.route('/admin')
def admin_dashboard():
    # Check if admin is logged in
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    conn = sqlite3.connect('transactions.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM transactions
        WHERE result='Fraud Detected' AND alerted=0
        ORDER BY id DESC
    """)
    alerts = cursor.fetchall()
    conn.close()

    return render_template('admin_dashboard.html', alerts=alerts)




@app.route('/acknowledge/<int:transaction_id>')
def acknowledge(transaction_id):
    conn = sqlite3.connect('transactions.db')
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE transactions
        SET alerted = 1
        WHERE id = ?
    """, (transaction_id,))
    conn.commit()
    conn.close()
    return "OK"


@app.route('/new_alerts_count')
def new_alerts_count():
    conn = sqlite3.connect('transactions.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE result='Fraud Detected' AND alerted=0")
    count = cursor.fetchone()[0]
    conn.close()
    return {"count": count}





# Hardcoded admin credentials for now
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == "admin" and password == "admin123":
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin_login.html', error="Invalid username or password")
    return render_template('admin_login.html')

@app.route('/admin_logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
