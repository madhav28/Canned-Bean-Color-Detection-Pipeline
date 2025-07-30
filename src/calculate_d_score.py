import pandas as pd
import numpy as np

def calculate_d_score():
    df = pd.read_csv("./output/results.csv")
    df['D_Score'] = np.sqrt(df['L_calib']**2+df['A_calib']**2+df['B_calib']**2)
    df['D_Score'] = 1-(df['D_Score']-12.75)/(22.5-12.75)
    df.to_csv("./output/results.csv", index=False)