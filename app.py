from flask import Flask, render_template, send_from_directory
import pandas as pd
import sqlite3
import os

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
    return render_template('index.html', table=df.to_html(classes='table table-striped table-bordered table-hover', index=False))

@app.route('/chart1')
def chart1():
    return render_template('chart.html', chart='1. Bar plot_Total Population by continent.png')

@app.route('/chart2')
def chart2():
    return render_template('chart.html', chart='2. Histogram_Population Growth Rates.png')

@app.route('/chart3')
def chart3():
    return render_template('chart.html', chart='3. Pie chart_Population Categories.png')

@app.route('/chart4')
def chart4():
    return render_template('chart.html', chart='4. Scatter Plot_Population vs Density.png')

@app.route('/chart5')
def chart5():
    return render_template('chart.html', chart='5. Scatter Plot_Population vs Growth Rate.png')

@app.route('/Charts/<path:filename>')
def send_chart(filename):
    return send_from_directory('Charts', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
