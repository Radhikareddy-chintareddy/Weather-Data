## IA626_Fall2023_FinalProjectRadhikaTejaswini
# WEATHER AND AIR POLLUTION DATA NY
## Weather Data:

 Weather data analysis plays a crucial role in understanding various environmental factors and their impacts on atmospheric conditions. This document aims to shed light on the complexities involved in analyzing weather data, particularly the relationships between temperature, pollutant levels, and visibility.
 

* 2022 weather data is used. In total there are 13474 csv files and each csv represents a particular station where the hourly data ia collected for the year 2022 from the stations located in various parts of the world. 
* Separating the US csv's from the rest of the csv's.
* In total there are 2729 us stations, and they were separated by the state of New York stations from US stations.
* There are 52 csv's for the NY state and joined all the csv's to get NY weather data as 'merged_data.csv'vwhich has 831932 rows and 16 columns.
* Given below is the code snipped for the above analysis.


```

source_directory = r'----------'  
us_data_directory = r'----\\weather\\us_data_NY' 

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
```


## Weather Data Overview:

### Data Sources:

Weather data originates from multiple sources such as NCEI SURFACE HOURLY/ASOS/AWOS, AFCCC’s USAF SURFACE HOURLY, or directly from USAF SURFACE HOURLY.
Various observation types (e.g., aerological, automatic stations, climate networks) provide diverse datasets.

The data has 

**Control Data Section -** it provides information about the report including date, time, and station location information. 

 **Mandatory Data Section -** The mandatory data section contains meteorological information on the basic elements such as wind, ceiling height (CIG), visibility (VIS), air temperature (TMP), dew point (DEW), sea level pressure (SLP). 

**Additional Data Section -** Variable length data are provided after the mandatory data. These additional data contain information of significance and/or which are received with varying degrees of frequency.

**Missing Values -** Missing values for any non-signed item are filled (i.e., 999). Missing values for any signed item are positive filled (i.e., +99999).

**Longitude and Latitude Coordinates -** Longitudes will be reported with negative values representing longitudes west of 0 degrees, and latiudes will be negative south of the equator.

Control, mandatory, and additional data sections contain different meteorological information with fixed or variable lengths. For the purpose of this project, only mandatory and control data sections are taken and columns with additional data section's are dropped.

Given Below are the first few rows of the data


| STATION      | DATE               | SOURCE | LATITUDE  | LONGITUDE  | ELEVATION | NAME                                | REPORT_TYPE | CALL_SIGN | QUALITY_CONTROL | WND            | CIG            | VIS           | TMP        | DEW        | SLP      |
|--------------|--------------------|--------|-----------|------------|-----------|-------------------------------------|-------------|-----------|-----------------|----------------|----------------|---------------|------------|------------|----------|
| 72055399999  | 2022-02-01T16:56:00| 4      | 40.701214 | -74.009028 | 2.13      | PORT AUTH DOWNTN MANHATTAN WALL ST HEL, NY US | FM-15       | 99999     | V020            | 060,1,N,0046,1 | 22000,1,9,N    | 016093,1,9,9  | +0022,1    | -0056,1    | 99999,9  |
| 72055399999  | 2022-02-01T17:56:00| 4      | 40.701214 | -74.009028 | 2.13      | PORT AUTH DOWNTN MANHATTAN WALL ST HEL, NY US | FM-15       | 99999     | V020            | 050,1,N,0036,1 | 22000,1,9,N    | 016093,1,9,9  | +0033,1    | -0056,1    | 99999,9  |


### Printing few unique station name in NY state.

```
unique_station_names = df['NAME'].unique()

print("Unique Station Names:")

for name in unique_station_names:

    print(name)
```


- LAGUARDIA AIRPORT, NY US
- POUGHKEEPSIE AIRPORT, NY US
- WESTCHESTER CO AIRPORT, NY US
- STEWART FIELD, NY US
- ISLIP LI MACARTHUR AIRPORT, NY US
- NY CITY CENTRAL PARK, NY US

The weather data for NY state is being sent to the database to establish an API. Through the development of a Flask application, the aim is to enable data retrieval based on specified start and end dates. This functionality will allow users to access requested weather data within their selected date range.

Given below are the code snippets and URL for the same:

 http://127.0.0.1:5000/getData?key=123&start=2022-02-01%2016:56:00&end=2022-02-01%2020:56:00

```
with conn.cursor() as cur:
       
        sql_create_table = '''
        CREATE TABLE IF NOT EXISTS `ny_weather` (
          `id` int NOT NULL AUTO_INCREMENT,
          `STATION` varchar(50) NOT NULL,
            -------)

cur.execute(sql_create_table)
        
        
        sql_insert_data = '''
        INSERT INTO `ny_weather` (`STATION`,`DATE`,`SOURCE`,`LATITUDE`,
            `LONGITUDE`,`ELEVATION`,`NAME`,`REPORT_TYPE`,`CALL_SIGN`,
            `QUALITY_CONTROL`,`WND`,`CIG`,`VIS`,`TMP`,`DEW`,`SLP`) 
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
        '''

```
```
@app.route("/getData", methods=['GET', 'POST'])
def getData():
    res = {} 
    res['req'] = '/getData'
    key = request.args.get('key')
    if key is None or key != '123':

    for row in cur:
        item = {
            'station_code': str(row['STATION']),
            'station_name': str(row['NAME']),
            'date_time': row['DATE'].strftime('%Y-%m-%d %H:%M:%S'),
        --------

        }
        items.append(item)

    res['results'] = items
    res['time_taken'] = execution_time
    
    return json.dumps(res, indent=4)

```

And given below is the result after a request :

```

{
    "req": "/getData",
    "code": 0,
    "msg": "ok",
    "results": [
        {
            "station_code": "72515594761",
            "station_name": "ITHACA TOMPKINS CNTY, NY US",
            "date_time": "2022-02-01 16:56:00",
            "data_source": "ASOS/AWOS observation merged with USAF SURFACE HOURLY observation",
            "Wind": "150,5,N,0088,5",
            "Temperature": "-0017,5",
            "Sea Level Pressure": "10303,5",
            "Ceiling Height": "22000,5,9,N",
            "Dew point": "-0117,5",
            "Visibility": "016093,5,N,5"
        },
        {
            "station_code": "72622794790",
            "station_name": "WATERTOWN INTERNATIONAL AIRPORT, NY US",
            "date_time": "2022-02-01 16:56:00",
            "data_source": "ASOS/AWOS observation merged with USAF SURFACE HOURLY observation",
            "Wind": "170,5,N,0057,5",
            "Temperature": "+0006,5",
            "Sea Level Pressure": "10290,5",
            "Ceiling Height": "22000,5,9,N",
            "Dew point": "-0122,5",
            "Visibility": "016093,5,N,5"
        },
```
## AIR Pollution Data:

The analysis focuses on four principal pollutants as defined by the Clean Air Act: carbon monoxide (CO), nitrogen dioxide (NO2), ground-level ozone (O3), and sulfur dioxide (SO2). These pollutants are key parameters evaluated to gauge air quality and potential environmental impacts.

The air pollution data was collected from the Environmental Protection Agency (EPA) via the EPA's Air Quality System (AQS). The dataset consists of pollutant data for ozone (O), nitrogen dioxide (NO2), sulfur dioxide (SO2), and carbon monoxide (CO) for the year 2022. The dataset contains hourly records with parameters such as latitude, longitude,State Code, County Code, Site Number, Sample Measurement, Parameter Code, date, and time.

## Objective:

The primary objective is to merge the air pollutant data with weather data for New York City. This merging is based on shared attributes including latitude, longitude, date, and time. The aim is to conduct a comparative analysis between air quality and weather conditions to ascertain any correlations or potential effects of air pollutants on temperature and visibility.

## Purpose of Analysis:

This analysis intends to examine potential relationships between air pollutants and weather conditions. It aims to explore if variations or changes in air pollutant levels have any discernible impact on temperature and visibility data obtained from the weather dataset. The data integration and analysis are critical to understanding the potential environmental influences of air quality on weather parameters.

All data used in this analysis is sourced from EPA's Air Quality System and is subject to their reporting standards and protocols.

## Merging the Weather and Air pollution datasets 

Only subset of the NY state is taken from the entire data after taking only necessary columns like parameter name, sample observations, Latitude, longitude, Date time and county state names and dropping the rest of the columns from all pollution csv's, Renaming the columns, merging the pollution datasets based on Latitude, longitude, county name, date time. 

Similarly subset of the weather Data is taken which has only NY state datapoints and these two data sets are merged on the common latitudes and longitudes.

```
# Find unique rounded latitude and longitude combinations in nyair_data
unique_nyair_lat_lon = ny_air[['Rounded_LAT', 'Rounded_LON']].drop_duplicates()

# Find unique rounded latitude and longitude combinations in weather_selected
unique_weather_lat_lon = weather[['Rounded_LAT', 'Rounded_LON']].drop_duplicates()

# Common rounded latitude and longitude values in both datasets
common_lat_lon = pd.merge(unique_nyair_lat_lon, unique_weather_lat_lon, on=['Rounded_LAT', 'Rounded_LON'], how='inner')

print(common_lat_lon)

unique_nyoz_values = nyair_subset[["County Name", "Rounded_LAT", "Rounded_LON"]].drop_duplicates()
print(unique_nyoz_values)

unique_weather_values = weather_subset[["NAME", "Rounded_LAT", "Rounded_LON"]].drop_duplicates()
unique_weather_values

```


       


| NAME                                       | Rounded_LAT | Rounded_LON |
|--------------------------------------------|-------------|-------------|
| PORT AUTH DOWNTN MANHATTAN WALL ST HEL, NY US | 41.0        | -74.0       |
| LAGUARDIA AIRPORT, NY US                   | 41.0        | -74.0       |
| WESTCHESTER CO AIRPORT, NY US              | 41.0        | -74.0       |
| NY CITY CENTRAL PARK, NY US                | 41.0        | -74.0       |
| JFK INTERNATIONAL AIRPORT, NY US           | 41.0        | -74.0       |
| THE BATTERY, NY US                         | 41.0        | -74.0       |
| KINGS POINT, NY US                         | 41.0        | -74.0       |


| County Name | Rounded_LAT | Rounded_LON |
|-------------|-------------|-------------|
| Bronx       | 41.0        | -74.0       |
| Queens      | 41.0        | -74.0       |


Given below is the first few rows after merging the NY subsets of airpollution data and weather data, cleaning the missing values and converting the temp to degrees celcius.

|        | 0          | 1          | 2          | 3          | 4               |
|--------|------------|------------|------------|------------|-----------------|
| Latitude              | 40.8679    | 40.8679    | 40.73614   | 40.73614   | 40.8679 |
| Longitude             | -73.87809  | -73.87809  | -73.82153  | -73.82153  | -73.87809 |
| Parameter Name_oz     | Ozone      | Ozone      | Ozone      | Ozone      | Ozone |
| Date Local            | 2022-01-04 | 2022-01-04 | 2022-01-04 | 2022-01-04 | 2022-01-04 |
| Time Local            | 11:00      | 11:00      | 11:00      | 11:00      | 12:00 |
| Sample Measurement_oz | 0.022      | 0.022      | 0.02       | 0.02       | 0.026 |
| Units of Measure_oz   | Parts per million | Parts per million | Parts per million | Parts per million | Parts per million |
| State Name            | New York   | New York   | New York   | New York   | New York |
| County Name           | Bronx      | Bronx      | Queens     | Queens     | Bronx |
| Parameter Name_co     | Carbon monoxide | Carbon monoxide | Carbon monoxide | Carbon monoxide | Carbon monoxide |
| Sample Measurement_co | 0.327      | 0.327      | 0.35       | 0.35       | 0.251 |
| Units of Measure_co   | Parts per million | Parts per million | Parts per million | Parts per million | Parts per million |
| Parameter Name_so     | Sulfur dioxide | Sulfur dioxide | Sulfur dioxide | Sulfur dioxide | Sulfur dioxide |
| Sample Measurement_so | 0.9        | 0.9        | 1.6        | 1.6        | 0.5 |
| Units of Measure_so   | Parts per billion | Parts per billion | Parts per billion | Parts per billion | Parts per billion |
| Parameter Name_no     | Nitrogen dioxide (NO2) | Nitrogen dioxide (NO2) | Nitrogen dioxide (NO2) | Nitrogen dioxide (NO2) | Nitrogen dioxide (NO2) |
| Sample Measurement_no | 17.1       | 17.1       | 22.2       | 22.2       | 14.4 |
| Units of Measure_no   | Parts per billion | Parts per billion | Parts per billion | Parts per billion | Parts per billion |
| STATION               | 99727199999 | 99728099999 | 99727199999 | 99728099999 | 72503014732 |
| DATE                  | 2022-01-04 11:00:00 | 2022-01-04 11:00:00 | 2022-01-04 11:00:00 | 2022-01-04 11:00:00 | 2022-01-04 12:00:00 |
| SOURCE                | 4          | 4          | 4          | 4          | 4 |
| ELEVATION             | 10.0       | 10.0       | 10.0       | 10.0       | 3.0 |
| NAME                  | THE BATTERY | KINGS POINT | THE BATTERY | KINGS POINT | LAGUARDIA AIRPORT |
| REPORT_TYPE           | FM-13      | FM-13      | FM-13      | FM-13      | FM-12 |
| CALL_SIGN             | NaN        | NaN        | NaN        | NaN        | NaN |
| QUALITY_CONTROL       | V020       | V020       | V020       | V020       | V020 |
| WND                   | NaN        | 010        | NaN        | 010        | 320 |
| CIG                   | NaN        | NaN        | NaN        | NaN        | NaN |
| VIS                   | NaN        | NaN        | NaN        | NaN        | 016000 |
| TMP                   | -67.0      | -64.0      | -67.0      | -64.0      | -61.0 |
| DEW                   | NaN        | NaN        | NaN        | NaN        | -0133 |
| SLP                   | 10296      | 10301      | 10296      | 10301      | 10298 |
| TMP_Celsius           | -6.7       | -6.4       | -6.7       | -6.4       | -6.1 |


The merged dataset has been analyzed to compute the average temperature across various counties and stations.

### Average Temperature by County

Bronx: 13.89°C

Queens: 12.99°C

### Average Temperature by Station Name

JFK INTERNATIONAL AIRPORT: 13.48°C

KINGS POINT: 12.93°C

LAGUARDIA AIRPORT: 14.31°C

NY CITY CENTRAL PARK: 10.96°C

PORT AUTH DOWNTN MANHATTAN WALL ST HEL: 16.00°C

THE BATTERY: 13.64°C

WESTCHESTER CO AIRPORT: 13.31°C

Subsequently, comprehensive time plots were generated, to check hourly and monthly trends for each parameter. These plots served as visual aids, enabling observation and comparison of patterns to examine potential correlations between weather conditions and air quality.

Given Below is the comparision of average ozone and CO levels for counties of NY.

![Alt text](<boxplot for ozone and co-1.png>)

This shows that the average remains almost same for both the counties.

Next a pairplot shows the relation between visibility and pollutants

![Alt text](<Visibility and airpollutants scatter plots-1.png>)


The plot appears inconclusive regarding the relationship between visibility and air pollutants. Maybe due to dataset's small and not able to capture trends, looks like datapoints are scattered across the plot, obscuring distinct relationships between visibility and pollution levels. maybe other factors such as varying weather conditions like fog or mist, are significantly impacting visibility. The relationship itself might not be linear. Anomalies, missing values, or outliers within the dataset might distort the true relationship between visibility and air pollutants, affecting the plot's accuracy and clarity. 

Given below is the scatter plot for temparature and pollutants

![Alt text](<temparature and airpollutants comparision plot-1.png>)

No relationship between temperature and pollutants except for ozone. Pollutants' concentrations are influenced by various factors like industrial emissions, traffic, wind patterns, and geographic location. Temperature might not be the sole influential factor on their concentration levels.

 There might be missing data or outliers impacting the analysis. Missing or incomplete data can influence the observations and interpretations.

 Given below are the hourly and monthly plots og pollutants and weather observations

 ![Alt text](<monthly weather data plots-1.png>) 
 
 ![Alt text](<hourly airpollution plots-1.png>) 
 
 ![Alt text](<hourly weather data plots-1.png>)
 
  ![Alt text](<monthly airpollution plots-1.png>)

Upon reviewing the hourly weather data depicted in the graphs, discernible patterns emerge. Ozone levels peak during midday hours, while CO, SO2, and NO2 exhibit higher concentrations in the mornings and evenings, showing a distinct correlation with daily traffic patterns. 

The surge in pollutant levels between 8:00 am to 10:00 am and 6:00 pm to 8:00 pm aligns with increased vehicular activity in New York City, indicating a probable link between traffic density and elevated pollutant levels.

Analysis of the monthly pollutants trend highlights that ozone levels remain notably elevated from May to September. This pattern strongly suggests a correlation between temperature and ozone, as the warmer summer months experience higher ozone concentrations. Conversely, other pollutants exhibit diminished levels during summer, presenting an inverse relationship compared to ozone.

Furthermore, the examination of hourly and monthly weather data illustrates distinct observations. Temperature and SLP (Sea Level Pressure) display peak values during the afternoon, while parameters like VIS (Visibility), WND (Wind), and DEW (Dew Point) do not follow discernible patterns, indicating a lack of clear trends in their variations over time.

Finally, the merged air and weather data for NY city has been pushed to a database, facilitating ease of access and management. An API has been successfully developed, enabling users to retrieve specific data by inputting a date. This API functionality serves as a convenient tool, allowing seamless extraction of relevant information corresponding to the provided date, enhancing accessibility and usability for data analysis and research purposes.

Given below are the code snippets:

http://127.0.0.1:5000/getDataNyairweather?key=123&date=2022-01-04%2011:00:00



```
with pymysql.connect(
     
) as conn:
 
    with conn.cursor() as cur:
        sql = '''DROP TABLE IF EXISTS `nyair_weather`'''
        cur.execute(sql)
       
        sql_create_table = '''
        CREATE TABLE IF NOT EXISTS `nyair_weather` (
            `id` int NOT NULL AUTO_INCREMENT,
            `Latitude` decimal(12,8) NOT NULL,
            `Longitude` decimal(12,8) NOT NULL,
            
            `TMP_Celsius` decimal(12,8) NOT NULL,
            PRIMARY KEY (`id`)
        ) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 AUTO_INCREMENT=1 COLLATE=utf8mb4_0900_ai_ci;
        ''' 
        cur.execute(sql_create_table)

```

```
@app.route("/getDataNyairweather", methods=['GET', 'POST'])
def getDataNyairweather():
    res = {} 
    res['req'] = '/getDataNyairweather'
    key = request.args.get('key')
    if key is None or key != '123':
        res['code'] = 1
        res['msg'] = 'key is invalid'
        return json.dumps(res, indent=4)
    
    date = request.args.get('date')
    
    print("Received date:", date)
    
    if date is None:
        res['code'] = 2
        res['msg'] = 'date must be provided.'
        return json.dumps(res, indent=4)
```

```
{
    "req": "/getDataNyairweather",
    "code": 0,
    "msg": "ok",
    "results": [
        {
            "Latitude": "40.86790000",
            "Longitude": "-73.87809000",
            "Parameter Name_oz": "Ozone",
            "Sample Measurement_oz": "0.02200000",
            "Units of Measure_oz": "Parts per million",
            "State Name": "New York",
            "County Name": "Bronx",
            "Parameter Name_co": "Carbon monoxide",
            "Sample Measurement_co": "0.32700000",
            "Units of Measure_co": "Parts per million",
            "Parameter Name_so": "Sulfur dioxide",
            "Sample Measurement_so": "0.90000000",
            "Units of Measure_so": "Parts per billion",
            "Sample Measurement_no": "17.10000000",
            "Units of Measure_no": "Parts per billion",
            "STATION": "99727199999",
            "DATE": "2022-01-04 11:00:00",
            "NAME": "THE BATTERY",
            "Wind": 0,
            "Ceiling Height": 0,
            "Visibility": 0,
            "Dew point": 0,
            "Temperature": -6.7
        },
            ],
    "time_taken": 0.8606431484222412
}
```

## CONCLUSION

The airpollution and weather data journey started by collecting information about air quality and weather from different sources. Then, the data subset based on the common latitude and longitude was taken. The subset ha NY data consisting of Bronx and queens counties. made graphs and charts to see patterns and connections between things like air pollution and weather changes.

Its noticed that pollution levels sometimes went up when there was more traffic in the city. Also, during hotter months, there was more ozone in the air, but other pollutants went down during these times.

Finally, an API is created to get specific data like air pollution or weather data for that day. This was a journey of looking at information, finding connections, and making it easier for everyone to access important data or data subset.