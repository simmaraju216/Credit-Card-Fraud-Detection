Credit Card Fraud Detection with Flask and Machine Learning
===========================================================

> due to some problem in render, ML model may not work in Website but locally it works fine.

Project Overview
----------------

This project aims to detect fraudulent credit card transactions using machine learning algorithms. It includes a web application built with Flask where users can upload transaction data, which is then processed using machine learning models to detect fraud.

User accounts and profile details are stored in memory, so they reset whenever the server restarts.

Installation
------------

Prerequisites:
- Python 3.x installed on your system.
- Git installed to clone the repository.

Steps:
1. Clone the repository:

   git clone
   cd Credit_Card_Fraud_Detection_Website

3. Set up Python environment:

   python -m venv env
   
   source `env/bin/activate`   #On Windows use `env\Scripts\activate`

5. Install dependencies:

   pip install -r requirements.txt

6. Download the dataset:

   - The dataset used in this project is not included in the repository due to size constraints.(you should download from kaggle)

7. Run the application:

   python app.py or flask run

   The application should now be running locally. Access it at http://localhost:5000 in your web browser.

Project Structure
-----------------

- app.py: Flask application setup, routes, and machine learning model preprocessing.
- templates/: HTML templates for rendering frontend.
- static/: CSS stylesheets and other static files.
- data/: Directory to store the dataset (not included in repository).

Machine Learning Models
-----------------------

This project uses the following machine learning models from scikit-learn for fraud detection:

- Isolation Forest
- Support Vector Classifier (SVC)
- Logistic Regression

Libraries used include pandas, Flask, scikit-learn, matplotlib, and Werkzeug for form handling and uploads.

Contributing
------------

Contributions are welcome! Please fork the repository and create a pull request for any improvements or fixes.
