# Libraries
import os
import json
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from kaggle.api.kaggle_api_extended import KaggleApi
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# API credentials for Kaggle
with open('kaggle.json') as f:
    data = json.load(f)

os.environ['KAGGLE_USERNAME'] = data['username']
os.environ['KAGGLE_KEY'] = data['key']

# Show current working directory
print(os.getcwd())

# Initialize API
api = KaggleApi()
api.authenticate()

# Download file
api.dataset_download_file('iamsouravbanerjee/world-population-dataset', 'world_population.csv')

# Read data to pandas data frame
df = pd.read_csv('world_population.csv', sep=',')

print("World Population Dataframe:")
print(df.to_string())

# Convert object columns to string
string_columns = df.select_dtypes(include=['object']).columns
df[string_columns] = df[string_columns].astype('string')

print(df.dtypes)

# Round numerical columns
df['Density (per km²)'] = df['Density (per km²)'].round(2)
df['Growth Rate'] = df['Growth Rate'].round(2)

print("\nDataFrame with rounded columns:")
print(df[['Density (per km²)', 'Growth Rate']].head())

# Define the PopulationData class
class PopulationData:
    def __init__(self, dataframe):
        self.df = dataframe

    def categorize_population(self, population):
        if population < 10_000_000:
            return 'Small'
        elif population < 25_000_000:
            return 'Medium'
        else:
            return 'Large'

    def add_population_category(self):
        self.df['Population Category'] = self.df['2022 Population'].apply(self.categorize_population)
        print("\nDataFrame with Population Category:")
        print(self.df[['Country/Territory', '2022 Population', 'Population Category']].head())

    def calculate_continent_summary(self):
        self.continent_population_summary = {}
        for continent in self.df['Continent'].unique():
            continent_data = self.df[self.df['Continent'] == continent]
            total_population = continent_data['2022 Population'].sum()
            average_population = int(np.ceil(continent_data['2022 Population'].mean()))  # Round up and convert to int
            self.continent_population_summary[continent] = {
                'Total Population': total_population,
                'Average Population': average_population
            }
        print("\nContinent Population Summary:")
        for continent, summary in self.continent_population_summary.items():
            print(f"{continent}: Total Population = {summary['Total Population']}, Average Population = {summary['Average Population']}")

    def filter_countries_by_population(self, threshold):
        filtered_countries = []
        for index, row in self.df.iterrows():
            if row['2022 Population'] > threshold:
                filtered_countries.append(row['Country/Territory'])
        print(f"\nCountries with population greater than {threshold}:")
        print(filtered_countries)
        return filtered_countries

    def display_table(self):
        print("\nTabulate DataFrame:")
        print(tabulate(self.df.head(), headers='keys', tablefmt='pretty'))

    def plot_population_categories(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='Population Category')
        plt.title('Number of Countries by Population Category')
        plt.xlabel('Population Category')
        plt.ylabel('Number of Countries')
        plt.show()

    def plot_continent_population_summary(self):
        continent_df = pd.DataFrame.from_dict(self.continent_population_summary, orient='index')
        continent_df.reset_index(inplace=True)
        continent_df.columns = ['Continent', 'Total Population', 'Average Population']

        plt.figure(figsize=(12, 6))
        sns.barplot(data=continent_df, x='Continent', y='Total Population')
        plt.title('Total Population by Continent')
        plt.xlabel('Continent')
        plt.ylabel('Total Population')
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.barplot(data=continent_df, x='Continent', y='Average Population')
        plt.title('Average Population by Continent')
        plt.xlabel('Continent')
        plt.ylabel('Average Population')
        plt.show()

    def plot_population_growth(self):
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=self.df, x='Country/Territory', y='Growth Rate')
        plt.title('Population Growth Rate by Country')
        plt.xlabel('Country/Territory')
        plt.ylabel('Growth Rate')
        plt.xticks(rotation=90)
        plt.show()

# Initialize the PopulationData class with the DataFrame
population_data = PopulationData(df)

# Add population category
population_data.add_population_category()

# Display table
population_data.display_table()

# Calculate continent population summary
population_data.calculate_continent_summary()

# Filter countries by population threshold
population_threshold = 100_000_000
filtered_countries = population_data.filter_countries_by_population(population_threshold)

# Plot population categories
population_data.plot_population_categories()

# Plot continent population summary
population_data.plot_continent_population_summary()

# Plot population growth
population_data.plot_population_growth()
