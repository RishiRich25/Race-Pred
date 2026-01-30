import fastf1 as ff1
import pandas as pd
from datetime import datetime as dt
from math import isnan
import csv




'''
ses = ff1.get_session(2018,4,4)
ses.load()
session = pd.DataFrame(ses.results)
session = session[['DriverId','Q1','Q2','Q3']]
session.set_index('DriverId',inplace=True)
x = session.loc['hamilton']['Q1']
print(type(x))


x1 = x['Q1']
seconds = x1.total_seconds()
if isnan(seconds):
    print("yes")
print(x1)
print(seconds)



events1 = ff1.get_event(2025,1)['EventName']
events2 = ff1.get_event(2024,3)['EventName']
print(events1==events2)


x = open('history_driver.csv','r')
reader = csv.reader(x)
data = []
for line in reader:
    if line != []:
        data.append(line)

df = pd.DataFrame(data)
print(df)
x.close()
'''
ff1.Cache.enable_cache('cache')
yr = 2025
schedule = ff1.get_event_schedule(yr, include_testing=False)
for rac in range(len(schedule)): 
    event = schedule.iloc[rac]
    event_format = event['EventFormat']
    rnd = event['RoundNumber']
    if event_format == "sprint" or event_format == "sprint_shootout" or event_format == "sprint_qualifying":
        ses = ff1.get_session(yr,rnd,'S')
        ses.load()
        wh = ses.weather_data
        result = ses.results
    ses = ff1.get_session(yr,rnd,'R')
    ses.load()
    wh = ses.weather_data
    result = ses.results
    quali = ff1.get_session(yr,rnd,'Q')
    quali.load()
    speeds = quali.results

