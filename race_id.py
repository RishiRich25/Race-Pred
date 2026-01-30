import pandas as pd
import csv
race_id = 0
with open("history_race.csv",'r') as file:
    data = csv.DictReader(file)
    races = []
    for row in data:
        try:
            if races[-1]['Track'] != row['Track']:
                race_id +=1
        except:
            pass
        row['Race_Id'] = race_id
        races.append(row)

with open("history_race.csv",'w') as file:
    writer = csv.writer(file)
    writer.writerow(['Driver','Team','Q1','Q2','Q3',
                     'Start','Finish','Track','Rain',
                     'D_Elo','T_Elo','Race_Id'])
    for race in races:
        row = [race['Driver'],race['Team'],race['Q1']
               ,race['Q2'],race['Q3'],race['Start']
               ,race['Finish'],race['Track'],race['Rain']
               ,race['D_Elo'],race['T_Elo'],race['Race_Id']]
        writer.writerow(row)