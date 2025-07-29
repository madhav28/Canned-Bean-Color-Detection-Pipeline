import pandas as pd

def calculate_d_score():
    df = pd.read_csv("./output/results.csv")
    df['D_Score'] = df['L_calib']**2+df['A_calib']**2+df['B_calib']**2
    df['D_Score'] = 1-(df['D_Score']-100)/(600-100)
    df.to_csv("./output/results.csv", index=False)