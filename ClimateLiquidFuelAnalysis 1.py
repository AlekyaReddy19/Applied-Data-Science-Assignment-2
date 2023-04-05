# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 21:48:43 2023

@author: Alekya
"""

# Libraries to be used
import pandas as pd

# FUNCTION THAT WILL RETURN TWO ARGUMENTS: YEARS AS COLUMNS AND COUNTRIES AS COLUMNS
def process_co2_data(filename):
    """Process CO2 data from a given CSV file and returns two dataframes:
    one with years as columns and another with countries as columns.

    Args:
        filename (str): The name of the CSV file.

    Returns:
        tuple: A tuple containing two dataframes: one with years as columns and
            another with countries as columns.
    """
    # Load the data into a DataFrame
    df = pd.read_csv(filename, skiprows=4)
    
    # Extract the data for the years 2010-2019
    yearsDF = df.loc[:, 'Country Name':'2019']
    yearsDF.columns = [col if not col.isdigit() else str(col) for col in yearsDF.columns]
    
    # Transpose the DataFrame to get a country-centric view
    countriesDF = yearsDF.transpose()
    
    # Replace empty values with 0
    countriesDF = countriesDF.fillna(0)
    
    # Set the column names for the countries DataFrame
    countriesDF.columns = countriesDF.iloc[0]
    countriesDF = countriesDF.iloc[1:]
    countriesDF.index.name = 'Year'
    
    # Set the column names for the years DataFrame
    yearsDF = yearsDF.rename(columns={'Country Name': 'Year'})
    yearsDF = yearsDF.set_index('Year')
    
    return yearsDF, countriesDF



#calling the function we created above
yearsDF, countriesDF = process_co2_data('API_EN.ATM.CO2E.LF.KT_DS2_en_csv_v2_4904068.csv')

# MAKING THE COUNTRIES TO BE COLUMNS
countriesDF

yearsDF

countriesDF.corr()

countriesDF.describe()

yearsDF.describe()

yearsDF.corr()

yearsDF.nlargest(50, '2016')

# A LINE CHART SHOWING THE `` OF 10 COUNTRIES
# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Reading the CSV file
df = pd.read_csv('API_EN.ATM.CO2E.LF.KT_DS2_en_csv_v2_4904068.csv', skiprows=4)

# Selecting the columns with the country names and the years 2002 to 2012
columns_to_select = ['Country Name'] + [f"{year}" for year in range(2007, 2017)]
df = df[columns_to_select]

# Filtering the dataframe to only include the 7 countries of your choice
countries_of_interest = ['United States', 'China', 'India', 'Japan', 'Russian Federation', 'Saudi Arabia', 'Brazil', 'Mexico', 'Canada', 'Germany']
df = df[df['Country Name'].isin(countries_of_interest)]

# Setting the index of the dataframe to be the 'Country Name' column
df.set_index('Country Name', inplace=True)

# Plotting the line chart
plt.figure(figsize=(10,6))
for country in countries_of_interest:
    plt.plot(df.loc[country], label=country)
plt.legend()
plt.title('Total greenhouse gas emissions (% change from 1990)')
plt.xlabel('Year')
plt.ylabel('% change from 1990')
plt.show()

#bar chart visualization of the top ten countries
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('API_EN.ATM.CO2E.LF.KT_DS2_en_csv_v2_4904068.csv', skiprows=4)

# Select the columns we want
columns = ['Country Name', '2012']
df = df[columns]

# Select the 7 countries of your choice
countries =  ['United States', 'China', 'India', 'Japan', 'Russian Federation', 'Saudi Arabia', 'Brazil', 'Mexico', 'Canada', 'Germany']
df = df[df['Country Name'].isin(countries)]

# Set the index to the country names
df.set_index('Country Name', inplace=True)

# Plot the bar chart
plt.figure(figsize=(12,6))
plt.bar(df.index, df['2012'])
plt.title('Total greenhouse gas emissions (% change from 1990) in 2012')
plt.xlabel('Country')
plt.ylabel('Total greenhouse gas emissions (% change from 1990)')
plt.show()

# PLOTTING THE HEATMAP FROM 2005 - 2015
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('API_EN.ATM.CO2E.LF.KT_DS2_en_csv_v2_4904068.csv', skiprows=4)

# Define a list of 10 countries of your choice
countries = ['United States', 'China', 'India', 'Japan', 'Russian Federation', 'Saudi Arabia', 'Brazil', 'Mexico', 'Canada', 'Germany']

# Filter the data for the 7 countries of your choice and the years 2002-2012
df_filtered = df.loc[df['Country Name'].isin(countries), ['Country Name', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014','2015']]
df_filtered.set_index('Country Name', inplace=True)

# Create a heatmap
plt.figure(figsize=(10, 8))
plt.imshow(df_filtered, cmap='YlOrBr', aspect='auto')
plt.xticks(range(len(df_filtered.columns)), df_filtered.columns)
plt.yticks(range(len(df_filtered.index)), df_filtered.index)
plt.colorbar()
plt.title('Total greenhouse gas emissions (% change from 1990)')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file and select the relevant columns
df = pd.read_csv('API_EN.ATM.CO2E.LF.KT_DS2_en_csv_v2_4904068.csv', skiprows=4)
df = df[['Country Name', '2016']]  # Select columns for country name and 2016 emissions

# Get the top countries in a list
top_countries = ['United States', 'China', 'India', 'Japan', 'Russian Federation', 'Saudi Arabia', 'Brazil', 'Mexico', 'Canada', 'Germany']  # Example list
df_top = df[df['Country Name'].isin(top_countries)]  # Filter the dataframe for the top countries

# Create the pie chart
plt.pie(df_top['2016'], labels=df_top['Country Name'], autopct='%1.1f%%')

# Add title
plt.title('CO2 Emissions from Liquid Fuel Consumption (kt) in Top Countries (2016)')

# Show the chart
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file and select the relevant columns
df = pd.read_csv('API_EN.ATM.CO2E.LF.KT_DS2_en_csv_v2_4904068.csv', skiprows=4)
df = df.iloc[:, [0] + list(range(50, 59))]  # Country Name and columns for years 2007-2016

# Define a list of countries you want to plot the area chart for
countries = ['United States', 'China', 'India', 'Japan', 'Russian Federation', 'Saudi Arabia', 'Brazil', 'Mexico', 'Canada', 'Germany']

# Subset the dataframe to include only the selected countries
top_10 = df[df['Country Name'].isin(countries)]

# Calculate the total emissions for each country over the 10 years
top_10['Total Emissions'] = top_10.iloc[:, 1:].sum(axis=1)

# Sort the data by total emissions
top_10 = top_10.sort_values('Total Emissions', ascending=False)

# Create the area chart
plt.stackplot(range(2007, 2017), top_10.iloc[:, 1:-1].T, labels=top_10['Country Name'])

# Add axis labels, legend, and title
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (kt)')
plt.title('Top 10 Countries with Highest CO2 Emissions from Liquid Fuel Consumption (2007-2016)')
plt.legend(loc='upper left')

# Show the chart
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('API_EN.ATM.CO2E.LF.KT_DS2_en_csv_v2_4904068.csv', skiprows=4)

# Select the relevant columns and rows
df_countries = df.loc[df['Country Name'].isin(['United States', 'China', 'India', 'Japan', 'Russian Federation', 'Saudi Arabia', 'Brazil', 'Mexico', 'Canada', 'Germany']), ['Country Name', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011']]


# Set the index to be the country names
df_countries.set_index('Country Name', inplace=True)

# Set the figure size and create a new subplot
plt.figure(figsize=(10, 6))
ax = plt.subplot()

# Set the years and the number of bars per group
years = ['2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011']
n_bars = len(years)

# Set the bar width and the offset between the groups
bar_width = 0.8 / n_bars
offset = bar_width / 2

# Set the colors for each year
colors = ['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c', '#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c']

# Set the x ticks to be the country names
x_ticks = df_countries.index

# Plot the bars for each year
for i, year in enumerate(years):
    ax.bar([j + offset + bar_width*i for j in range(len(x_ticks))], df_countries[year], width=bar_width, label=year, color=colors[i])

# Set the axis labels and title
ax.set_xlabel('Country')
ax.set_ylabel('CO2 Emissions (kt)')
ax.set_title('CO2 Emissions by Country and Year')

# Set the x ticks and labels
ax.set_xticks([j + 0.4 for j in range(len(x_ticks))])
ax.set_xticklabels(x_ticks, rotation=60)

# Add a legend
ax.legend()

# Show the plot
plt.show()




