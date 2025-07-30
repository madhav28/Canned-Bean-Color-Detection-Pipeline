import pandas as pd
import numpy as np
import json

def calculate_d_score():
    with open('./input/class.txt', 'r') as f:
        class_name = f.readline().strip()

    with open('./assets/config.json') as f:
        class_config = json.load(f)

    L = class_config[class_name]["L"]
    A = class_config[class_name]["A"]
    B = class_config[class_name]["B"]
    D_Score_Min = class_config[class_name]["D Score Min"]
    D_Score_Max = class_config[class_name]["D Score Max"]

    df = pd.read_csv("./output/results.csv")
    df['D_Score'] = np.sqrt((df['L_calib']-L)**2+(df['A_calib']-A)**2+(df['B_calib']-B)**2)
    df['D_Score'] = 1-(df['D_Score']-D_Score_Min)/(D_Score_Max-D_Score_Min)
    cols_to_round = ['L_initial', 'A_initial', 'B_initial', 'L_calib', 'A_calib', 'B_calib']
    df[cols_to_round] = df[cols_to_round].round(2)
    df['D_Score'] = (df['D_Score'].clip(0, 1) * 10).astype(int) + 1
    df['D_Score'] = df['D_Score'].clip(1, 10)
    df.to_csv("./output/results.csv", index=False)