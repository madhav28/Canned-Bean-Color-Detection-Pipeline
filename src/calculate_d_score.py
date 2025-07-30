import pandas as pd
import numpy as np

def calculate_d_score():
    df = pd.read_csv("./output/results.csv")
    df['D_Score'] = np.sqrt(df['L_calib']**2+df['A_calib']**2+df['B_calib']**2)
    df['D_Score'] = 1-(df['D_Score']-12.75)/(22.5-12.75)
    cols_to_round = ['L_initial', 'A_initial', 'B_initial', 'L_calib', 'A_calib', 'B_calib']
    df[cols_to_round] = df[cols_to_round].round(2)
    df['D_Score'] = (df['D_Score'].clip(0, 1) * 10).astype(int) + 1
    df['D_Score'] = df['D_Score'].clip(1, 10)
    df.to_csv("./output/results.csv", index=False)