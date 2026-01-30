import fastf1 as ff1
import pandas as pd
from datetime import datetime as dt
import csv
import time
import gc
ff1.Cache.enable_cache('cache')
def driver_elo_calc_past(elo, start_pos, finish_pos, status, k_fact, grid=19):
    start_pos = int(start_pos)
    expected = 1 / (1 + 10 ** ((start_pos - 10) / 10))
    if finish_pos.isdigit():
        finish_pos = int(finish_pos)
    if status in ['Finished','Lapped',
    '+1 Lap', '+2 Laps', '+3 Laps', '+4 Laps', '+5 Laps',
    '+6 Laps', '+7 Laps', '+8 Laps', '+9 Laps', '+10 Laps',
    '+11 Laps', '+12 Laps', '+13 Laps', '+14 Laps', '+15 Laps',
    '+16 Laps', '+17 Laps', '+18 Laps', '+19 Laps', '+20 Laps',
    '+21 Laps', '+22 Laps', '+23 Laps', '+24 Laps', '+25 Laps',
    '+26 Laps', '+27 Laps', '+28 Laps', '+29 Laps', '+30 Laps',
    '+31 Laps', '+32 Laps', '+33 Laps', '+34 Laps', '+35 Laps',
    '+36 Laps', '+37 Laps', '+38 Laps', '+39 Laps', '+40 Laps',
    '+41 Laps', '+42 Laps', '+43 Laps', '+44 Laps', '+45 Laps',
    '+46 Laps', '+47 Laps', '+48 Laps', '+49 Laps', '+50 Laps',
    '+51 Laps', '+52 Laps', '+53 Laps', '+54 Laps', '+55 Laps',
    '+56 Laps', '+57 Laps', '+58 Laps', '+59 Laps']:
        actual = max(0, 1 - (finish_pos - 1) / grid)
    elif status in ["DNF","DSQ","DQ",'Accident','Collision','Collision damage','Damage','Retired']:
        actual = 0.0
    else:
        return elo
    elo = float(elo)
    actual = float(actual)
    expected = float(expected)
    return round(elo + k_fact * (actual - expected), 2)


def team_elo_calc_past(elo, start_pos, finish_pos, status, k_fact, grid=19):
    expected = 1 / (1 + 10 ** ((start_pos - 10) / 10))
    start_pos = int(start_pos)
    if finish_pos.isdigit():
        finish_pos = int(finish_pos)
    if status in ['Finished','Lapped',
    '+1 Lap', '+2 Laps', '+3 Laps', '+4 Laps', '+5 Laps',
    '+6 Laps', '+7 Laps', '+8 Laps', '+9 Laps', '+10 Laps',
    '+11 Laps', '+12 Laps', '+13 Laps', '+14 Laps', '+15 Laps',
    '+16 Laps', '+17 Laps', '+18 Laps', '+19 Laps', '+20 Laps',
    '+21 Laps', '+22 Laps', '+23 Laps', '+24 Laps', '+25 Laps',
    '+26 Laps', '+27 Laps', '+28 Laps', '+29 Laps', '+30 Laps',
    '+31 Laps', '+32 Laps', '+33 Laps', '+34 Laps', '+35 Laps',
    '+36 Laps', '+37 Laps', '+38 Laps', '+39 Laps', '+40 Laps',
    '+41 Laps', '+42 Laps', '+43 Laps', '+44 Laps', '+45 Laps',
    '+46 Laps', '+47 Laps', '+48 Laps', '+49 Laps', '+50 Laps',
    '+51 Laps', '+52 Laps', '+53 Laps', '+54 Laps', '+55 Laps',
    '+56 Laps', '+57 Laps', '+58 Laps', '+59 Laps']:
        actual = max(0, 1 - (finish_pos - 1) / grid)
    elif status in ["DNS","DSQ","DQ",'Suspension','Engine','Gearbox',
                    'Transmission','Brakes','Electrical','Hydraulics',
                    'Power Unit','Fuel pressure','Oil leak','Oil pressure',
                    'Water pressure','Cooling','Driveshaft','Steering',
                    'Clutch','Exhaust','Fuel system','Throttle','Battery'
                    ,'ERS','MGU-H','MGU-K','Turbo','Mechanical','Technical','Retired']:
        actual = 0.0
    else:
        return elo
    elo = float(elo)
    actual = float(actual)
    expected = float(expected)
    return round(elo + k_fact * (actual - expected), 2)


driver_data = {"Name":[],
               "Elo":[]}

with open('history_driver.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)
    driver_data_list = []
    for row in csv_reader:
        driver_data_list.append(row)

for data in driver_data_list:
    driver_data["Name"].append(data["Name"])
    driver_data["Elo"].append(data["Elo"])


team_data = {"Name":[],
             "Elo":[]}

with open('history_team.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)
    team_data_list = []
    for row in csv_reader:
        team_data_list.append(row)

for data in team_data_list:
    team_data["Name"].append(data["Name"])
    team_data["Elo"].append(data["Elo"])

race_stats = {"Driver":[],
              "Team":[],
              "Q1":[],
              "Q2":[],
              "Q3":[],
              "Start":[],
              "Finish":[],
              "Track":[],
              "Rain":[],
              "D_Elo":[],
              "T_Elo":[]}

#load new data at the end of each year
yr = 2025
schedule = ff1.get_event_schedule(yr, include_testing=False)
for rac in range(len(schedule)):
    try:
        event = schedule.iloc[rac]
        track = event['EventName']
        event_format = event['EventFormat']
        rnd = event['RoundNumber']
        k_fact_driver = 0
        k_fact_team = 0
        if event_format == "sprint" or event_format == "sprint_shootout" or event_format == "sprint_qualifying":

            ses = ff1.get_session(yr,rnd,'S')
            ses.load()
                
            wh = ses.weather_data
            wh = wh[["Rainfall"]]
            rain = wh['Rainfall'].any()

            
            if rain:
                k_fact_driver = 16
                k_fact_team = 12
            else:
                k_fact_driver = 12
                k_fact_team = 8

            session = pd.DataFrame(ses.results)
            session = session[['DriverId','TeamId','ClassifiedPosition','GridPosition','Status']]
            grid = len(session)
            for dr in range(grid):
                try:
                    driver = session.iloc[dr]

                    if driver['DriverId'] not in driver_data["Name"]:
                        driver_data["Name"].append(driver['DriverId'])
                        driver_data["Elo"].append(1200)
                    else:
                        pos = driver_data["Name"].index(driver['DriverId'])
                        temp_elo = driver_data["Elo"][pos]
                        temp_elo = driver_elo_calc_past(temp_elo, driver['GridPosition'], driver['ClassifiedPosition'], driver['Status'],k_fact_driver, grid-1)
                        driver_data["Elo"][pos] = temp_elo

                    if driver['TeamId'] not in team_data["Name"]:
                        team_data["Name"].append(driver['TeamId'])
                        team_data["Elo"].append(1800)
                    else:
                        pos = team_data["Name"].index(driver['TeamId'])
                        temp_elo = team_data["Elo"][pos]
                        temp_elo = team_elo_calc_past(temp_elo, driver['GridPosition'], driver['ClassifiedPosition'], driver['Status'],k_fact_team, grid-1)
                        team_data["Elo"][pos] = temp_elo
                except:
                    pass
            del ses
            gc.collect()


        ses = ff1.get_session(yr,rnd,'R')
        ses.load()
            
        wh = ses.weather_data
        wh = wh[["Rainfall"]]
        rain = wh['Rainfall'].any()


        if rain:
            k_fact_driver = 24
            k_fact_team = 18
        else:
            k_fact_driver = 20
            k_fact_team = 15

        session = pd.DataFrame(ses.results)
        session = session[['DriverId','TeamId','ClassifiedPosition','GridPosition','Status']]
        quali = ff1.get_session(yr,rnd,'Q')
        quali.load()
        quali_times = pd.DataFrame(quali.results)
        quali_times = quali_times[['DriverId','Q1','Q2','Q3']]
        quali_times.set_index('DriverId',inplace=True)
        grid = len(session)
        for dr in range(grid):
            try:
                driver = session.iloc[dr]
                if driver['DriverId'] not in driver_data["Name"]:
                    driver_data["Name"].append(driver['DriverId'])
                    driver_data["Elo"].append(1200)
                else:
                    pos = driver_data["Name"].index(driver['DriverId'])
                    temp_elo = driver_data["Elo"][pos]
                    temp_elo = driver_elo_calc_past(temp_elo, driver['GridPosition'], driver['ClassifiedPosition'], driver['Status'],k_fact_driver, grid-1)
                    driver_data["Elo"][pos] = temp_elo

                q1 = quali_times.loc[driver['DriverId']]['Q1']
                q2 = quali_times.loc[driver['DriverId']]['Q2']
                q3 = quali_times.loc[driver['DriverId']]['Q3']
                race_stats['Driver'].append(driver['DriverId'])
                race_stats['Q1'].append(q1.total_seconds())
                race_stats['Q2'].append(q2.total_seconds())
                race_stats['Q3'].append(q3.total_seconds())
                race_stats['Start'].append(driver['GridPosition'])
                if driver['ClassifiedPosition'].isdigit():
                    race_stats['Finish'].append(float(driver['ClassifiedPosition']))
                else:
                    race_stats['Finish'].append(float('nan'))
                race_stats['Rain'].append(rain)
                race_stats['Track'].append(track)
                race_stats['D_Elo'].append(temp_elo)

                
                if driver['TeamId'] not in team_data["Name"]:
                    team_data["Name"].append(driver['TeamId'])
                    team_data["Elo"].append(1800)
                else:
                    pos = team_data["Name"].index(driver['TeamId'])
                    temp_elo = team_data["Elo"][pos]
                    temp_elo = team_elo_calc_past(temp_elo, driver['GridPosition'], driver['ClassifiedPosition'], driver['Status'],k_fact_team, grid-1)
                    team_data["Elo"][pos] = temp_elo
                
                race_stats['Team'].append(driver['TeamId'])
                race_stats['T_Elo'].append(temp_elo)
            except:
                pass
        del ses
        del quali
        gc.collect()
    except:
        pass


rows_1 = []
for i in range(len(driver_data["Name"])):
    row = [driver_data["Name"][i], driver_data["Elo"][i]]
    rows_1.append(row)

rows_2 = []
for i in range(len(team_data["Name"])):
    row = [team_data["Name"][i], team_data["Elo"][i]]
    rows_2.append(row)

filename_3 = "history_race.csv"
csvfile_3 = open(filename_3,'a')
csvwriter_3 = csv.writer(csvfile_3)
for i in range(len(race_stats['Driver'])):
    row = [race_stats['Driver'][i],race_stats['Team'][i],race_stats['Q1'][i],race_stats['Q2'][i],race_stats['Q3'][i],
           race_stats['Start'][i],race_stats['Finish'][i],race_stats['Track'][i],race_stats['Rain'][i]
           ,race_stats['D_Elo'][i],race_stats['T_Elo'][i]]
    csvwriter_3.writerow(row)
csvfile_3.close()


filename_1 = "history_driver.csv"
with open(filename_1, 'w') as csvfile_1:
    csvwriter_1 = csv.writer(csvfile_1)
    csvwriter_1.writerow(["Name","Elo"])
    csvwriter_1.writerows(rows_1)

filename_2 = "history_team.csv"
with open(filename_2, 'w') as csvfile_2:
    csvwriter_2 = csv.writer(csvfile_2)
    csvwriter_2.writerow(["Name","Elo"])
    csvwriter_2.writerows(rows_2)
