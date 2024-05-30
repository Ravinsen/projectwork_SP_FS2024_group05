from flask import Flask, render_template
import pandas as pd
import sqlite3

app = Flask(__name__)

# Function to get data from the database
def get_data():
    conn = sqlite3.connect('world_population_DB.db')
    df = pd.read_sql_query("SELECT * FROM population", conn)
    conn.close()
    return df

@app.route('/')
def index():
    df = get_data()
    return render_template('index.html', tables=[df.to_html(classes='data')], titles=df.columns.values)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
