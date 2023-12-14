#  
import csv, pymysql,time


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
from flask import Flask, request, jsonify
import pymysql
import json

app = Flask(__name__)


@app.route("/getData", methods=['GET', 'POST'])
def getData():
    res = {} 
    res['req'] = '/getData'
    key = request.args.get('key')
    if key is None or key != '123':
        res['code'] = 1
        res['msg'] = 'key is invalid'
        return json.dumps(res, indent=4)
    
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    print("Received start_date:", start_date)
    print("Received end_date:", end_date)
    if start_date is None or end_date is None:
        res['code'] = 2
        res['msg'] = 'Both start and end dates must be provided.'
        return json.dumps(res, indent=4)
    
    conn = pymysql.connect(host='**********', port=3306, user='chintar', passwd='***********', db='chintar_Bigdataweather', autocommit=True) 
    cur = conn.cursor(pymysql.cursors.DictCursor)
    
    start_time = time.time()
    sql = 'SELECT * FROM `ny_weather` WHERE `DATE` BETWEEN %s AND %s ORDER BY `DATE` LIMIT 0, 500;'
    cur.execute(sql, (start_date, end_date))
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    res['code'] = 0
    res['msg'] = 'ok'
    items = []
    
    for row in cur:
        item = {
            'station_code': str(row['STATION']),
            'station_name': str(row['NAME']),
            'date_time': row['DATE'].strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': decode_source(str(row['SOURCE'])),
            'Wind': row['WND'],
            'Temperature': row['TMP'],
            'Sea Level Pressure': row['SLP'],
            'Ceiling Height': row['CIG'],
            'Dew point': row['DEW'],
            'Visibility': row['VIS'],

        }
        items.append(item)

    res['results'] = items
    res['time_taken'] = execution_time
    
    return json.dumps(res, indent=4)


import decimal
def decimal_default(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

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
    
    conn = pymysql.connect(host='************', port=3306, user='chintar', passwd='**********', db='chintar_Bigdataweather', autocommit=True) 
    cur = conn.cursor(pymysql.cursors.DictCursor)
    
    start_time = time.time()
    sql = 'SELECT * FROM `nyair_weather` WHERE `DATE` = %s LIMIT 0, 500;'
    cur.execute(sql, date)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    res['code'] = 0
    res['msg'] = 'ok'
    items = []
    
    for row in cur:
        item = {
            'Latitude': str(row['Latitude']),
            'Longitude': str(row['Longitude']),
            'Parameter Name_oz': str(row['Parameter Name_oz']),
            'Sample Measurement_oz': str(row['Sample Measurement_oz']),
            'Units of Measure_oz': str(row['Units of Measure_oz']),
            'State Name': str(row['State Name']),
            'County Name': str(row['County Name']),
            'Parameter Name_co': str(row['Parameter Name_co']),
            'Sample Measurement_co': str(row['Sample Measurement_co']),
            'Units of Measure_co': str(row['Units of Measure_co']),
            'Parameter Name_so': str(row['Parameter Name_so']),
            'Sample Measurement_so': str(row['Sample Measurement_so']),
            'Units of Measure_so': str(row['Units of Measure_so']),
            'Sample Measurement_no': str(row['Sample Measurement_no']),
            'Units of Measure_no': str(row['Units of Measure_no']),
            'STATION': str(row['STATION']),
            'DATE': row['DATE'].strftime('%Y-%m-%d %H:%M:%S'),
            'NAME': str(row['NAME']),
            'Wind': row['WND'],
            'Ceiling Height': row['CIG'],
            'Visibility': row['VIS'],
            'Dew point': row['DEW'],   
            'Temperature': float(row['TMP_Celsius']),  
        }

        items.append(item)

    res['results'] = items
    res['time_taken'] = execution_time
    
 
    json_response = json.dumps(res, default=decimal_default, indent=4)
    return json_response

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)


