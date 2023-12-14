# %%
import pandas as pd
import os


directory = r'C:\------\Final project\weather\2022'


csv_files = [filename for filename in os.listdir(directory) if filename.endswith(".csv")]


for i, filename in enumerate(csv_files):
    file_path = os.path.join(directory, filename)
    
    try:
       
        df = pd.read_csv(file_path, delimiter='\t')  
        
       
        if i % 1000 == 0:
            print(f"Column names in {filename}: {list(df.columns)}")
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
    
    i += 1  


# %%
#iterating through the csv's of the weather data, keeping the columns upto SLP and dropping the remaining columns.
# It has taken one hour to drop the columns.


# %%
import pandas as pd
import os


directory_path = r'C:\-------\Final project\weather\2022'


file_list = [filename for filename in os.listdir(directory_path) if filename.endswith('.csv')]


columns_to_keep = [
    'STATION', 'DATE', 'SOURCE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME',
    'REPORT_TYPE', 'CALL_SIGN', 'QUALITY_CONTROL', 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP'
]


for file_name in file_list:
    file_path = os.path.join(directory_path, file_name)
   
    df = pd.read_csv(file_path)
   
    columns_to_drop = [col for col in df.columns if col not in columns_to_keep]
    df.drop(columns=columns_to_drop, inplace=True)
 
    df.to_csv(file_path, index=False)


# %%
#I have taken the 2022 data. In total there are 13474 csv files and each csv represents a particular station where the hourly data ia collected for the year 2022. 
#Separating the US csv's from the rest of the csv's.

# %%
import os
import pandas as pd
import shutil

source_directory = 'C:\\---------\\Final project\\weather\\2022'  
us_data_directory = 'C:\\------\\Final project\\weather\\us_data_full'  


if not os.path.exists(us_data_directory):
    os.makedirs(us_data_directory)


files = os.listdir(source_directory)
for file in files:
    if file.endswith('.csv'):  
        file_path = os.path.join(source_directory, file)
        
        
        df = pd.read_csv(file_path, usecols=['NAME'])
        df['NAME'] = df['NAME'].astype(str)
        
        
        us_rows = df[df['NAME'].str.contains(r', [A-Z]{2} US$', na=False)]
        
     
        if not us_rows.empty:
            
            destination_path = os.path.join(us_data_directory, file)
            shutil.move(file_path, destination_path)
            print(f"Moved {file} to {destination_path}")


# %%
import os
import pandas as pd
import shutil

source_directory = 'C:\\-------\\Final project\\weather\\2022' 
us_data_directory = 'C:\\------\\weather\\us_data_full'  

if not os.path.exists(us_data_directory):
    os.makedirs(us_data_directory)


files = os.listdir(source_directory)
for file in files:
    if file.endswith('.csv'): 
        file_path = os.path.join(source_directory, file)
        
       
        df = pd.read_csv(file_path, usecols=['NAME'])
        df['NAME'] = df['NAME'].astype(str)
     
        us_rows = df[df['NAME'].str.contains(r', [A-Z]{2} US$', na=False)]
        
      
        if not us_rows.empty:
            
            destination_path = os.path.join(us_data_directory, file)
            shutil.move(file_path, destination_path)
            print(f"Moved {file} to {destination_path}")


# %%
# In total there are 2729 us stations, and I have separated the state of New York stations from US stations.

# %%
import os
import pandas as pd
import shutil

source_directory = r'C:\\-----\Final project\\weather\\us_data_full'  
us_data_directory = r'C:\\-------\\Final project\\weather\\us_data_NY' 

if not os.path.exists(us_data_directory):
    os.makedirs(us_data_directory)


files = os.listdir(source_directory)
for file in files:
    if file.endswith('.csv'):  
        file_path = os.path.join(source_directory, file)
        
      
        df = pd.read_csv(file_path, usecols=['NAME'])
        df['NAME'] = df['NAME'].astype(str)
        
      
        us_rows = df[df['NAME'].str.contains(r', \bNY\b US$', na=False)]
        
        
        if not us_rows.empty:
            
            destination_path = os.path.join(us_data_directory, file)
            shutil.move(file_path, destination_path)
            print(f"Moved {file} to {destination_path}")


# %%
#here I have got 52 csv's for the NY state and I have joined all the csv's to get NY weather data.

# %%
import os
import pandas as pd


directory = r'C:\-----\Final project\weather\us_data_NY'

files = os.listdir(directory)


dataframes = []


for file in files:
    if file.endswith('.csv'):  
        filepath = os.path.join(directory, file)
        df = pd.read_csv(filepath)
        dataframes.append(df)


merged_df = pd.concat(dataframes)


merged_df.to_csv('merged_data.csv', index=False)  


# %%
merged_df.shape

# %%

# Printing the unique station name in NY state.

df = pd.read_csv('merged_data.csv')


unique_station_names = df['NAME'].unique()
print("Unique Station Names:")
for name in unique_station_names:
    print(name)


# %%
merged_df.nunique()

# %%
merged_df.columns

# %%
merged_df['REPORT_TYPE'].unique()

# %%
merged_df['CALL_SIGN'].unique()

# %%
merged_df.info()

# %%
# I am sending the data to the database to create an API.

# %%
import csv
import pymysql
import time


with open('merged_data.csv') as f:
    data = [{k: str(v) for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]


with pymysql.connect(
        host='***************',
        port=3306,
        user='chintar',
        passwd='*********',
        db='chintar_Bigdataweather',
        autocommit=True
) as conn:
 
    with conn.cursor() as cur:
       
        sql_create_table = '''
        CREATE TABLE IF NOT EXISTS `ny_weather` (
          `id` int NOT NULL AUTO_INCREMENT,
          `STATION` varchar(50) NOT NULL,
          `DATE` datetime NOT NULL,
          `SOURCE` varchar(20) NOT NULL,
          `LATITUDE` decimal(12,8) NOT NULL,
          `LONGITUDE` decimal(12,8) NOT NULL,
          `ELEVATION` decimal(12,8) NOT NULL,
          `NAME`  varchar(200)NOT NULL,
          `REPORT_TYPE` varchar(20) NOT NULL,
          `CALL_SIGN` varchar(20) NOT NULL,
          `QUALITY_CONTROL` varchar(20) NOT NULL,
          `WND` varchar(50) NOT NULL,
          `CIG` varchar(50) NOT NULL,
          `VIS` varchar(50) NOT NULL,
          `TMP` varchar(50) NOT NULL,
          `DEW` varchar(50) NOT NULL,
          `SLP` varchar(50) NOT NULL,
           PRIMARY KEY (`id`)
        ) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 AUTO_INCREMENT=1 COLLATE=utf8mb4_0900_ai_ci;
        '''
       
        cur.execute(sql_create_table)
        
        
        sql_insert_data = '''
        INSERT INTO `ny_weather` (`STATION`,`DATE`,`SOURCE`,`LATITUDE`,
            `LONGITUDE`,`ELEVATION`,`NAME`,`REPORT_TYPE`,`CALL_SIGN`,
            `QUALITY_CONTROL`,`WND`,`CIG`,`VIS`,`TMP`,`DEW`,`SLP`) 
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
        '''

        
        blocksize = 1000
        tokens = []
        n = 0
        dur = 0

       
        for row in data:
            tokens.append([
                row['STATION'], row['DATE'], row['SOURCE'], row['LATITUDE'], row['LONGITUDE'],
                row['ELEVATION'], row['NAME'], row['REPORT_TYPE'], row['CALL_SIGN'],
                row['QUALITY_CONTROL'], row['WND'], row['CIG'], row['VIS'], row['TMP'], row['DEW'], row['SLP']
            ])
            if len(tokens) >= blocksize:
                start = time.time()
                try:
                    
                  
                    cur.executemany(sql_insert_data, tokens)
                    conn.commit()
                    dur += time.time() - start
                    tokens = []
                    print(n)
                except Exception as e:
                    print(f"Error: {e}")
                    conn.rollback()
                n += 1

     
        if len(tokens) > 0:
            start = time.time()
            try:
                cur.executemany(sql_insert_data, tokens)
                dur += time.time() - start
            except Exception as e:
                print(f"Error: {e}")
                conn.rollback()

        print(dur)


# %%
merged_df['SOURCE'].unique()

# %%
#creating a flask application so that I can get the data based on start and end dates
#http://127.0.0.1:5000/getData?key=123&start=2022-02-01%2016:56:00&end=2022-02-01%2020:56:00

# %%
import json
from flask import Flask
from flask import request, redirect

from datetime import datetime

# %%
def decode_source(value):
    sources = {
        '4': 'USAF SURFACE HOURLY observation',
        '7': 'ASOS/AWOS observation merged with USAF SURFACE HOURLY observation',
        'O': 'Summary observation created by NCEI using hourly observations that may not share the same data source flag.',
        '6': 'ASOS/AWOS observation from NCEI',  
        '1': 'USAF SURFACE HOURLY observation, candidate for merge with NCEI SURFACE HOURLY (not yet merged, element cross-checks)',
        '2': 'NCEI SURFACE HOURLY observation, candidate for merge with USAF SURFACE HOURLY (not yet merged, failed element cross-checks)',
        'I': 'Climate Reference Network observation'
        
    }
    return sources.get(str(value), 'Unknown')


decoded_source = decode_source(4)  
print(decoded_source)

# %%
app = Flask(__name__)

###Get Data


@app.route("/getData", methods=['GET','POST'])
def getData():
    res = {} 
    res['req'] = '/getData'
    key = request.args.get('key')
    if key is None or key != '123':
        res['code'] = 1
        res['msg'] = 'key is invalid'
        return json.dumps(res,indent=4)
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    if start_date is None or end_date is None:
        res['code'] = 2
        res['msg'] = 'Both start and end dates must be provided.'
        return json.dumps(res, indent=4)
    conn = pymysql.connect(host='*************', port=3306, user='chintar',passwd='*********', db='chintar_Bigdataweather', autocommit=True) 
    cur = conn.cursor(pymysql.cursors.DictCursor)
    start_time = time.time()
    sql = 'SELECT * FROM `ny_weather` WHERE `DATE` BETWEEN %s AND %s ORDER BY `DATE` LIMIT 0,500;'
    cur.execute(sql,(start_date, end_date))
    end_time = time.time()
    execution_time = end_time - start_time
    res['code'] = 0
    res['msg'] = 'ok'
    items = []
    for row in cur:
        item = {}
        row['station_code'] = str(row['STATION'])
        row['date_time'] = row['DATE'].strftime('%Y-%m-%d %H:%M:%S')
        row['data_source'] = decode_source(str(row['SOURCE']))
        items.append(item)

    res['results'] = items
    res['time_taken'] = execution_time
    return json.dumps(res, indent=4)

if __name__ == '__main__':
    app.run(host='127.0.0.1',debug=True)


# %%
'''Now I have airpollution data that I have collected from EPA i.e environmental protection agency epa.org. 
I have collected the data of 2022 for pollutants such as ozone (o), Nitrogen dioxide (No2), sodium dioxide (so2), And carbon monoxide (co)
 the idea is to merge the pollutants data based on latitude, longitude, and Date and time for the 
weather and air pollution data of NY city to see if I can campare those two data sets.
To check if there is any effect of airpollutants on temparature and visibility obtained from the weather data.'''


# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)
import matplotlib.patches as mpatches

%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

# %%
# air pollution data

# %%
oz_2022 = pd.read_csv("hourly_44201_2022.csv")
co_2022 = pd.read_csv("hourly_42101_2022.csv")
no_2022 = pd.read_csv("hourly_42602_2022.csv")
so_2022 = pd.read_csv("hourly_42401_2022.csv")

# %%
oz_2022.columns

# %%
# Taking only necessary columns like parameter name, sample observations, Latitude, longitude, Date time and county state names
 #and dropping the rest of the columns from all pollution csv's. Renaming the columns.
#merging the pollution datasets based on Latitude, longitude, county name, date time.

# %%
columns_to_drop = ['State Code', 'County Code', 'Site Num', 'Parameter Code', 'POC',
        'Datum',  'Date GMT', 'Time GMT',  'MDL', 'Uncertainty', 'Qualifier', 'Method Type',
       'Method Code', 'Method Name',
       'Date of Last Change', ]

oz_2022 = oz_2022.drop(columns=columns_to_drop)

# %%
columns_to_exclude = ['Latitude', 'Longitude', 'Date Local', 'Time Local', 'County Name', 'State Name']

for col in oz_2022.columns:
    if col not in columns_to_exclude:
        oz_2022 = oz_2022.rename(columns={col: col + '_oz'})

# %%
oz_2022.head(2)

# %%
columns_to_drop = ['State Code', 'County Code', 'Site Num', 'Parameter Code', 'POC',
       'Datum',  'Date GMT', 'Time GMT',  'MDL', 'Uncertainty', 'Qualifier', 'Method Type',
       'Method Code', 'Method Name',
       'Date of Last Change', ]

co_2022 = co_2022.drop(columns=columns_to_drop)

# %%
columns_to_drop = ['State Code', 'County Code', 'Site Num', 'Parameter Code', 'POC',
       'Datum',  'Date GMT', 'Time GMT',  'MDL', 'Uncertainty', 'Qualifier', 'Method Type',
       'Method Code', 'Method Name',
       'Date of Last Change', ]

so_2022 = so_2022.drop(columns=columns_to_drop)

# %%
columns_to_drop = ['State Code', 'County Code', 'Site Num', 'Parameter Code', 'POC',
       'Datum',  'Date GMT', 'Time GMT',  'MDL', 'Uncertainty', 'Qualifier', 'Method Type',
       'Method Code', 'Method Name',
       'Date of Last Change', ]

no_2022 = no_2022.drop(columns=columns_to_drop)

# %%
for col in co_2022.columns:
    if col not in columns_to_exclude:
        co_2022 = co_2022.rename(columns={col: col + '_co'})

# %%
for col in so_2022.columns:
    if col not in columns_to_exclude:
        so_2022 = so_2022.rename(columns={col: col + '_so'})

# %%
for col in no_2022.columns:
    if col not in columns_to_exclude:
        no_2022 = no_2022.rename(columns={col: col + '_no'})

# %%
merged_df = pd.merge(oz_2022, co_2022, on=['Date Local', 'Time Local', 'County Name', 'State Name','Latitude','Longitude'])

# %%
merged_data = pd.merge(merged_df, so_2022, on=['Date Local', 'Time Local', 'County Name', 'State Name','Latitude','Longitude'])

# %%
air_data = pd.merge(merged_data, no_2022, on=['Date Local', 'Time Local', 'County Name', 'State Name','Latitude','Longitude'])

# %%
air_data.head().T

# %%
air_data.shape

# %%
#separate the New York city counties from the air data

# %%
ny_counties = air_data[air_data['State Name'] == 'New York']['County Name'].unique()

# Display the unique counties in New York State
for county in ny_counties:
    print(county)

# %%
ny_counties_list = ['Bronx', 'Queens']

# Filter 'ny_2022' dataset for New York counties
ny_air = air_data[air_data['County Name'].isin(ny_counties_list)]

# %%
ny_air.shape

# %%
# Now I have taken the weather data of the NY state that i have separated.
# I am first trying to compare the lat ans longitude values of the two datasets to see if i can find any common places in both datasets.
#to do that I am taking down the precision of lat and long values.
#Because weather data has station laitude and longitude.
#air data has, county lat and long which that station is nearby.

# %%
weather = pd.read_csv("merged_data.csv")

# %%
ny_air['Rounded_LAT'] = ny_air['Latitude'].round(0)
ny_air['Rounded_LON'] = ny_air['Longitude'].round(0)

# %%
weather['Rounded_LAT'] = weather['LATITUDE'].round(0)
weather['Rounded_LON'] = weather['LONGITUDE'].round(0)

# %%
# Find unique rounded latitude and longitude combinations in nyair_data
unique_nyair_lat_lon = ny_air[['Rounded_LAT', 'Rounded_LON']].drop_duplicates()

# Find unique rounded latitude and longitude combinations in weather_selected
unique_weather_lat_lon = weather[['Rounded_LAT', 'Rounded_LON']].drop_duplicates()

# Common rounded latitude and longitude values in both datasets
common_lat_lon = pd.merge(unique_nyair_lat_lon, unique_weather_lat_lon, on=['Rounded_LAT', 'Rounded_LON'], how='inner')

print(common_lat_lon)

# %%
# Common latitude and longitude combinations
common_lat_lon = [
    (41.0, -74.0)]

# Construct a boolean mask for filtering nyair_data
mask = ny_air.apply(lambda row: (row['Rounded_LAT'], row['Rounded_LON']) in common_lat_lon, axis=1)


nyair_subset = ny_air[mask]

# %%
unique_nyoz_values = nyair_subset[["County Name", "Rounded_LAT", "Rounded_LON"]].drop_duplicates()
print(unique_nyoz_values)

# %%
common_lat_lon = [
    (41.0, -74.0)]

# Construct a boolean mask for filtering weather data
mask = weather.apply(lambda row: (round(row['Rounded_LAT'], 1), round(row['Rounded_LON'], 1)) in common_lat_lon, axis=1)

# Apply the mask to weather data
weather_subset = weather[mask]

# %%
unique_weather_values = weather_subset[["NAME", "Rounded_LAT", "Rounded_LON"]].drop_duplicates()
unique_weather_values

# %%
weather_subset.shape

# %%
nyair_subset.shape

# %%
#it can be observe that weather data has 7 stations in the NY and air data has two counties that are matching.

# %%
weather_subset['DATE'] = pd.to_datetime(weather_subset['DATE'])

# Extracting date and time separately into new columns
weather_subset['Date'] = weather_subset['DATE'].dt.date  # Extract date
weather_subset['Time'] = weather_subset['DATE'].dt.time  # Extract time

# %%
nyair_subset['Datetime'] = pd.to_datetime(nyair_subset['Date Local'] + ' ' + nyair_subset['Time Local'])

# Convert 'Date' and 'Time' columns to datetime in weather data subset
weather_subset['Datetime'] = pd.to_datetime(weather_subset['Date'].astype(str) + ' ' + weather_subset['Time'].astype(str))

# Merge the DataFrames on the 'Datetime' column
merged_data = pd.merge(nyair_subset, weather_subset , on='Datetime', how='inner')

# Display the merged DataFrame
merged_data.head().T

# %%


# Columns to keep 
columns_to_keep = ['Latitude', 'Longitude', 'Parameter Name_oz', 'Date Local',
       'Time Local', 'Sample Measurement_oz', 'Units of Measure_oz',
       'State Name', 'County Name', 'Parameter Name_co',
       'Sample Measurement_co', 'Units of Measure_co', 'Parameter Name_so',
       'Sample Measurement_so', 'Units of Measure_so', 'Parameter Name_no',
       'Sample Measurement_no', 'Units of Measure_no', 'STATION', 'DATE', 'SOURCE', 'ELEVATION', 'NAME', 'REPORT_TYPE', 'CALL_SIGN',
       'QUALITY_CONTROL', 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP',
       ]

# Filter DataFrame to keep only relevant columns
data = merged_data[columns_to_keep]



# %%
data.head().T

# %%
data_final = data.astype(str)  # Convert all columns to strings
data_final = data_final.apply(lambda row: row.str.split(',').str[0])  # Extract first values after splitting


# %%
data_final[['WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP']].apply(lambda x: x.unique())


# %%
values_to_replace = ['999', '9999', '99999', '999999','-9999','+9999']

# Replace these values with NaN
data_final.replace(values_to_replace, np.nan, inplace=True)

# %%
import numpy as np


data_final['TMP'] = pd.to_numeric(data_final['TMP'], errors='coerce')


data_final['TMP'].replace(9999.0, np.nan, inplace=True)

# Convert temperature values to Celsius
data_final['TMP_Celsius'] = data_final['TMP'] / 10.0


print(data_final[['TMP', 'TMP_Celsius']])


# %%
#final data set after Merging
data_final.head().T

# %%
#avg temparature based on county
avg_temp_county = data_final.groupby('County Name')['TMP_Celsius'].mean()


print(avg_temp_county)

# %%
avg_temp_station = data_final.groupby('NAME')['TMP_Celsius'].mean()


print(avg_temp_station)

# %%
columns_to_convert = ['Sample Measurement_oz', 'Sample Measurement_co', 'Sample Measurement_so', 'Sample Measurement_no', 'WND', 'CIG', 'VIS', 'SLP']


data_final[columns_to_convert] = data_final[columns_to_convert].apply(pd.to_numeric, errors='coerce')

data_final.info()

# %%
data_final.columns

# %%
# Columns to keep 
columns_to_keep =['Latitude', 'Longitude', 'Parameter Name_oz', 'Date Local',
       'Time Local', 'Sample Measurement_oz', 'Units of Measure_oz',
       'State Name', 'County Name', 'Parameter Name_co',
       'Sample Measurement_co', 'Units of Measure_co', 'Parameter Name_so',
       'Sample Measurement_so', 'Units of Measure_so', 'Parameter Name_no',
       'Sample Measurement_no', 'Units of Measure_no', 'STATION', 'DATE',
       'NAME', 'WND', 'CIG', 'VIS', 'DEW', 'SLP',
       'TMP_Celsius']

# Filter DataFrame to keep only relevant columns
data_final = data_final[columns_to_keep]

# %%
# Combine 'Date Local' and 'Time Local' into a single DateTime column
data_final['DateTime'] = pd.to_datetime(data_final['Date Local'] + ' ' + data_final['Time Local'])

# Convert 'DateTime' column to DateTime format
data_final['DateTime'] = pd.to_datetime(data_final['DateTime'])

# Extract Hour and Month columns for further analysis
data_final['Hour'] = data_final['DateTime'].dt.hour
data_final['Month'] = data_final['DateTime'].dt.to_period('M')


# %%
data_final.head().T

# %%
import matplotlib.pyplot as plt

# Filter data for Queens and Bronx counties
queens_data = data_final[data_final['County Name'] == 'Queens']
bronx_data = data_final[data_final['County Name'] == 'Bronx']

# Compare average ozone levels between Queens and Bronx
queens_ozone_mean = queens_data['Sample Measurement_oz'].mean()
bronx_ozone_mean = bronx_data['Sample Measurement_oz'].mean()

# Compare average carbon monoxide levels between Queens and Bronx
queens_co_mean = queens_data['Sample Measurement_co'].mean()
bronx_co_mean = bronx_data['Sample Measurement_co'].mean()

# Visualize comparative analysis for ozone levels
plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)
plt.bar(['Queens', 'Bronx'], [queens_ozone_mean, bronx_ozone_mean], color=['skyblue', 'salmon'])
plt.xlabel('County')
plt.ylabel('Average Ozone Measurement')
plt.title('Comparison of Average Ozone Levels')
plt.grid(axis='y')

# Visualize comparative analysis for carbon monoxide levels
plt.subplot(1, 2, 2)
plt.bar(['Queens', 'Bronx'], [queens_co_mean, bronx_co_mean], color=['skyblue', 'salmon'])
plt.xlabel('County')
plt.ylabel('Average CO Measurement')
plt.title('Comparison of Average Carbon Monoxide Levels')
plt.grid(axis='y')

plt.tight_layout()
plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Select columns for comparison
columns_to_compare = ['VIS', 'Sample Measurement_oz', 'Sample Measurement_co', 'Sample Measurement_so', 'Sample Measurement_no']

# Create a pairplot (scatter matrix) for the selected columns
sns.pairplot(data_final[columns_to_compare], kind='scatter', diag_kind='hist')
plt.suptitle('Relationships between Visibility and Pollutants', y=1.02)
plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Select columns for comparison
columns_to_compare = ['TMP_Celsius', 'Sample Measurement_oz', 'Sample Measurement_co', 'Sample Measurement_so', 'Sample Measurement_no']

# Create a pairplot (scatter matrix) for the selected columns
sns.pairplot(data_final[columns_to_compare], kind='scatter', diag_kind='hist')
plt.suptitle('Relationships between Temparature and Pollutants', y=1.02)
plt.show()

# %%

hourly_oz_mean = data_final.groupby('Hour')['Sample Measurement_oz'].mean()
hourly_co_mean = data_final.groupby('Hour')['Sample Measurement_co'].mean()
hourly_so_mean = data_final.groupby('Hour')['Sample Measurement_so'].mean()
hourly_no2_mean = data_final.groupby('Hour')['Sample Measurement_no'].mean()



fig, axs = plt.subplots(4, 1, figsize=(10, 16))

# Ozone (O3)
axs[0].plot(hourly_oz_mean.index, hourly_oz_mean.values, label='Ozone', marker='o', color='blue')
axs[0].set_title('Average Ozone Measurement per Hour')
axs[0].set_xlabel('Hour of the Day')
axs[0].set_ylabel('Average Ozone Measurement')
axs[0].legend()
axs[0].grid(True)

# Carbon monoxide (CO)
axs[1].plot(hourly_co_mean.index, hourly_co_mean.values, label='CO', marker='o', color='green')
axs[1].set_title('Average Carbon Monoxide Measurement per Hour')
axs[1].set_xlabel('Hour of the Day')
axs[1].set_ylabel('Average CO Measurement')
axs[1].legend()
axs[1].grid(True)

# Sulfur dioxide (SO2)
axs[2].plot(hourly_so_mean.index, hourly_so_mean.values, label='SO2', marker='o', color='red')
axs[2].set_title('Average Sulfur Dioxide Measurement per Hour')
axs[2].set_xlabel('Hour of the Day')
axs[2].set_ylabel('Average SO2 Measurement')
axs[2].legend()
axs[2].grid(True)

# Nitrogen dioxide (NO2)
axs[3].plot(hourly_no2_mean.index, hourly_no2_mean.values, label='NO2', marker='o', color='orange')
axs[3].set_title('Average Nitrogen Dioxide Measurement per Hour')
axs[3].set_xlabel('Hour of the Day')
axs[3].set_ylabel('Average NO2 Measurement')
axs[3].legend()
axs[3].grid(True)


plt.tight_layout()
plt.show()




# %%

data_final['Date Local'] = pd.to_datetime(data_final['Date Local'])


data_final['Month'] = data_final['Date Local'].dt.to_period('M')

# Grouping by month and calculating mean for each pollutant
monthly_means = data_final.groupby('Month').agg({
    'Sample Measurement_oz': 'mean',
    'Sample Measurement_co': 'mean',
    'Sample Measurement_so': 'mean',
    'Sample Measurement_no': 'mean'
})


fig, axs = plt.subplots(4, 1, figsize=(10, 16))

# Ozone (O3)
axs[0].plot(monthly_means.index.to_timestamp(), monthly_means['Sample Measurement_oz'], label='Ozone', marker='o', color='blue')
axs[0].set_title('Average Ozone Measurement for Different Months')
axs[0].set_xlabel('Month')
axs[0].set_ylabel('Average Ozone Measurement')
axs[0].legend()
axs[0].grid(True)

# Carbon monoxide (CO)
axs[1].plot(monthly_means.index.to_timestamp(), monthly_means['Sample Measurement_co'], label='CO', marker='o', color='green')
axs[1].set_title('Average Carbon Monoxide Measurement for Different Months')
axs[1].set_xlabel('Month')
axs[1].set_ylabel('Average CO Measurement')
axs[1].legend()
axs[1].grid(True)

# Sulfur dioxide (SO2)
axs[2].plot(monthly_means.index.to_timestamp(), monthly_means['Sample Measurement_so'], label='SO2', marker='o', color='red')
axs[2].set_title('Average Sulfur Dioxide Measurement for Different Months')
axs[2].set_xlabel('Month')
axs[2].set_ylabel('Average SO2 Measurement')
axs[2].legend()
axs[2].grid(True)

# Nitrogen dioxide (NO2)
axs[3].plot(monthly_means.index.to_timestamp(), monthly_means['Sample Measurement_no'], label='NO2', marker='o', color='orange')
axs[3].set_title('Average Nitrogen Dioxide Measurement for Different Months')
axs[3].set_xlabel('Month')
axs[3].set_ylabel('Average NO2 Measurement')
axs[3].legend()
axs[3].grid(True)


plt.tight_layout()
plt.show()


# %%
columns_to_convert = ['TMP_Celsius', 'VIS', 'CIG', 'SLP', 'DEW']
data_final[columns_to_convert] = data_final[columns_to_convert].astype(float)

# Grouping by month and calculating mean for each weather parameter
monthly_weather_means = data_final.groupby('Month').agg({
    'TMP_Celsius': 'mean',
    'VIS': 'mean',
    'CIG': 'mean',
    'SLP': 'mean',
    'DEW': 'mean',
    'WND': 'mean'
})


fig, axs = plt.subplots(6, 1, figsize=(10, 20))

# Temperature (TMP)
axs[0].plot(monthly_weather_means.index.to_timestamp(), monthly_weather_means['TMP_Celsius'], label='Temperature', marker='o', color='purple')
axs[0].set_title('Average Temperature for Different Months')
axs[0].set_xlabel('Month')
axs[0].set_ylabel('Average Temperature (Celsius)')
axs[0].legend()
axs[0].grid(True)

# Visibility (VIS)
axs[1].plot(monthly_weather_means.index.to_timestamp(), monthly_weather_means['VIS'], label='Visibility', marker='o', color='green')
axs[1].set_title('Average Visibility for Different Months')
axs[1].set_xlabel('Month')
axs[1].set_ylabel('Average Visibility')
axs[1].legend()
axs[1].grid(True)

# Ceiling (CIG)
axs[2].plot(monthly_weather_means.index.to_timestamp(), monthly_weather_means['CIG'], label='Ceiling', marker='o', color='orange')
axs[2].set_title('Average Ceiling for Different Months')
axs[2].set_xlabel('Month')
axs[2].set_ylabel('Average Ceiling')
axs[2].legend()
axs[2].grid(True)

# Sea Level Pressure (SLP)
axs[3].plot(monthly_weather_means.index.to_timestamp(), monthly_weather_means['SLP'], label='Sea Level Pressure', marker='o', color='blue')
axs[3].set_title('Average Sea Level Pressure for Different Months')
axs[3].set_xlabel('Month')
axs[3].set_ylabel('Average SLP')
axs[3].legend()
axs[3].grid(True)

# Dew Point (DEW)
axs[4].plot(monthly_weather_means.index.to_timestamp(), monthly_weather_means['DEW'], label='Dew Point', marker='o', color='red')
axs[4].set_title('Average Dew Point for Different Months')
axs[4].set_xlabel('Month')
axs[4].set_ylabel('Average Dew Point')
axs[4].legend()
axs[4].grid(True)

# WND
axs[5].plot(monthly_weather_means.index.to_timestamp(), monthly_weather_means['DEW'], label='WND', marker='o', color='red')
axs[5].set_title('Average wind for Different Months')
axs[5].set_xlabel('Month')
axs[5].set_ylabel('Average WND')
axs[5].legend()
axs[5].grid(True)

plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt


data_final['Time Local'] = pd.to_timedelta(data_final['Time Local'] + ':00')  


data_final['Hour'] = data_final['Time Local'].dt.components['hours']

# Group by hour and calculate mean for each weather parameter
hourly_weather_means = data_final.groupby('Hour').agg({
    'TMP_Celsius': 'mean',
    'VIS': 'mean',
    'CIG': 'mean',
    'SLP': 'mean',
    'DEW': 'mean',
    'WND': 'mean'
})


fig, axs = plt.subplots(len(hourly_weather_means.columns), 1, figsize=(10, 16))


for i, col in enumerate(hourly_weather_means.columns):
    axs[i].plot(hourly_weather_means.index, hourly_weather_means[col], marker='o')
    axs[i].set_title(f'Hourly Average {col} Measurement')
    axs[i].set_xlabel('Hour of the Day')
    axs[i].set_ylabel(f'Average {col} Measurement')
    axs[i].grid(True)


plt.tight_layout()
plt.show()


# %%


# %%


# %%
import pandas as pd


columns_to_keep = ['Latitude', 'Longitude', 'Parameter Name_oz',  'Sample Measurement_oz', 'Units of Measure_oz',
       'State Name', 'County Name', 'Parameter Name_co',
       'Sample Measurement_co', 'Units of Measure_co', 'Parameter Name_so',
       'Sample Measurement_so', 'Units of Measure_so', 'Parameter Name_no',
       'Sample Measurement_no', 'Units of Measure_no', 'STATION', 'DATE',
       'NAME', 'WND', 'CIG', 'VIS', 'DEW', 'SLP', 'TMP_Celsius']


data_final = data_final[columns_to_keep]

# %%
data_final.to_csv('data_final.csv', index=False)

# %%
data_final.columns

# %%
#Sending the data to database

# %%
import csv
import pymysql
import time



with pymysql.connect(
        host='***********',
        port=3306,
        user='chintar',
        passwd='********',
        db='chintar_Bigdataweather',
        autocommit=True
) as conn:
 
    with conn.cursor() as cur:
        sql = '''DROP TABLE IF EXISTS `nyair_weather`'''
        cur.execute(sql)
       
        sql_create_table = '''
        CREATE TABLE IF NOT EXISTS `nyair_weather` (
            `id` int NOT NULL AUTO_INCREMENT,
            `Latitude` decimal(12,8) NOT NULL,
            `Longitude` decimal(12,8) NOT NULL,
            `Parameter Name_oz` varchar(200) NOT NULL,
            `Sample Measurement_oz` decimal(12,8) NOT NULL,
            `Units of Measure_oz` varchar(20) NOT NULL,
            `State Name` varchar(50) NOT NULL,
            `County Name` varchar(50) NOT NULL,
            `Parameter Name_co` varchar(200) NOT NULL,
            `Sample Measurement_co` decimal(12,8) NOT NULL,
            `Units of Measure_co` varchar(20) NOT NULL,
            `Parameter Name_so` varchar(200) NOT NULL,
            `Sample Measurement_so` decimal(12,8) NOT NULL,
            `Units of Measure_so` varchar(20) NOT NULL,
            `Sample Measurement_no` decimal(12,8) NOT NULL,
            `Units of Measure_no` varchar(20) NOT NULL,
            `STATION` varchar(20) NOT NULL,
            `DATE` datetime NOT NULL,
            `NAME` varchar(200) NOT NULL,
            `WND` INT NOT NULL,
            `CIG` INT NOT NULL,
            `VIS` INT NOT NULL,
            `DEW` INT NOT NULL,
            
            `TMP_Celsius` decimal(12,8) NOT NULL,
            PRIMARY KEY (`id`)
        ) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 AUTO_INCREMENT=1 COLLATE=utf8mb4_0900_ai_ci;
        '''
       
        cur.execute(sql_create_table)
        
        
        sql_insert_data = '''
            INSERT INTO `nyair_weather` (
                `Latitude`, `Longitude`, `Parameter Name_oz`, `Sample Measurement_oz`, `Units of Measure_oz`,
                `State Name`, `County Name`, `Parameter Name_co`, `Sample Measurement_co`, `Units of Measure_co`,
                `Parameter Name_so`, `Sample Measurement_so`, `Units of Measure_so`,
                `Sample Measurement_no`, `Units of Measure_no`, `STATION`, `DATE`, `NAME`, `WND`, `CIG`,
                `VIS`, `DEW`, `TMP_Celsius`
            ) 
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            );
        '''

        blocksize = 1000
        tokens = []
        n = 0
        dur = 0

        for row in data:
         
            wind_value = float(row['WND']) if row['WND'] else 0
            cig_value = float(row['CIG']) if row['CIG'] else 0
            
            dew_value = float(row['DEW']) if row['DEW'] else 0
            vis_value = float(row['VIS']) if row['VIS'] else 0
            tmp_celsius_value = float(row['TMP_Celsius']) if row['TMP_Celsius'] else 0

            tokens.append([
                row['Latitude'], row['Longitude'], row['Parameter Name_oz'], row['Sample Measurement_oz'], row['Units of Measure_oz'],
                row['State Name'], row['County Name'], row['Parameter Name_co'], row['Sample Measurement_co'], row['Units of Measure_co'],
                row['Parameter Name_so'], row['Sample Measurement_so'], row['Units of Measure_so'], row['Sample Measurement_no'],
                row['Units of Measure_no'], row['STATION'], row['DATE'], row['NAME'], wind_value, cig_value, vis_value, dew_value, tmp_celsius_value
                
            ])
            
            if len(tokens) >= blocksize:
                start = time.time()
                try:
                    cur.executemany(sql_insert_data, tokens)
                    conn.commit()
                    dur += time.time() - start
                    tokens = []
                    print(n)
                except Exception as e:
                    print(f"Error: {e}")
                    conn.rollback()
                n += 1

        if len(tokens) > 0:
            start = time.time()
            try:
                cur.executemany(sql_insert_data, tokens)
                dur += time.time() - start
            except Exception as e:
                print(f"Error: {e}")
                conn.rollback()

        print(dur)


# %%



