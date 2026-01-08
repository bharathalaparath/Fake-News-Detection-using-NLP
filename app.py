from flask import Flask, render_template, request
import joblib
import re
import string
import pandas as pd
import os

app = Flask(__name__)

# Get the absolute path to the directory containing this script
current_file_path = os.path.abspath(__file__)
Current_dir = os.path.dirname(__file__)
print(f"\nCurrent script path: {current_file_path}\nCurrent directory: {Current_dir}")

# Construct the path to Model.pkl in the parent directory
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
model_path = os.path.join(Current_dir, "model.pkl")
print(f"Corrected model path: {model_path}")

# os.path.dirname(__file__) — Returns the directory in which the current script (__file__) is located.
# os.pardir — Refers to the parent directory (usually "..").
# os.path.join(..., os.pardir) — Joins the current script’s directory with its parent, effectively stepping up one level.
# os.path.abspath(...) — Converts this parent directory path to its absolute form, which means you get the full directory path regardless of where the script is run.

# Load the model
Model = joblib.load(model_path)

# @app.route('/')
# def index():
#     return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])

def index():
    result = None
    if request.method == 'POST':
        txt = request.form.get('txt', '')
        processed_txt = wordpre(txt)
        data = pd.Series([processed_txt])
        try:
            result = Model.predict(data)[0]
        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template("index.html", result=result)

def wordpre(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)  # remove special characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# @app.route('/', methods=['POST'])
# def pre():
#     if request.method == 'POST':
#         txt = request.form['txt']
#         txt = wordpre(txt)
#         txt = pd.Series([txt])  # Ensure this is passed as a list or a Series
        
#         try:
#             result = Model.predict(txt)
#             result = result[0]  # Assuming it's a list/array, get the first prediction
#         except Exception as e:
#             result = f"Error: {str(e)}"
        
#         return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)