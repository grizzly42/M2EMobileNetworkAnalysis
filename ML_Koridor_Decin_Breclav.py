# Required imports are insert here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib.ticker import  MultipleLocator


# Type here location of your input files
files = {
    "T-Mobile": "dataset_Koridor_Decin_Breclav\\T_Mobile.csv",
    "O2": "dataset_Koridor_Decin_Breclav\\O2.csv",
    "Vodafone": "dataset_Koridor_Decin_Breclav\\Vodafone.csv"
}
# Color settings for operators
colors = {'T-Mobile': 'magenta','O2': 'blue','Vodafone': 'red'}

# Here define lacation whrere you want to save figures as PDF
SaveFigLoc = "ML_results"

models = {
    "Linear Regression": LinearRegression(),
    # 
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    # 
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42)
}

# Limit values
LevelMaxRSRP = -50
LevelMinRSRP = -130

# setting the size of the figures
FigSizeSet = (24,12)



# Setting fugure text style to LaTex font and font size
unitFontSize = 20
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "axes.titlepad": 24,
"axes.labelsize": unitFontSize, "axes.titlesize": unitFontSize, "xtick.labelsize": unitFontSize, "ytick.labelsize": unitFontSize, "legend.fontsize": unitFontSize, "figure.titlesize": unitFontSize})




## function that does data preprocessing
def processOP(file_path, operator_name):
    # loading the file
    df = pd.read_csv(file_path, low_memory=False)
    
    # Splitting the first column into date and time
    split_cols = df[df.columns[0]].astype(str).str.split('_', n=1, expand=True)
    split_cols.columns = ['date', 'time']
    split_cols['date'] = split_cols['date'].str.replace('.', '-', regex=False)
    split_cols['time'] = split_cols['time'].str.replace('.', ':', n=2, regex=False)
    df = pd.concat([split_cols, df.drop(columns=[df.columns[0]])], axis=1)
    
    # conversion to numbers
    df["Level"] = pd.to_numeric(df["Level"], errors="coerce")
    if "SNR" in df.columns:
        df["SNR"] = pd.to_numeric(df["SNR"], errors="coerce")
        
    # Filtering redundant data
    df =  df[df["EVENT"] == "PING TEST"]
    # Filtering irrelevant RSRP data
    df = df[(df["Level"] >= LevelMinRSRP) & (df["Level"] <= LevelMaxRSRP)]
    
    # Datetime
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    df = df.sort_values('datetime')
    
    # distance calculation using harvesin's formula
    lat_rad = np.radians(df["Latitude"].values)
    lon_rad = np.radians(df["Longitude"].values)
    R = 6371000
    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)
    a = np.sin(dlat/2)**2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = np.insert(R * c, 0, 0)
    # distance conversion to kilometers
    df['Distance_km'] = pd.Series(dist).cumsum() / 1000
    df['Operator'] = operator_name
    return df

# making output directory
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / SaveFigLoc 
OUTPUT_DIR.mkdir(exist_ok=True)


# Process each file store the resulting DataFrames
dataframes = [processOP(file, op) for op, file in files.items()]
# Concatenate all DataFrames into a single DataFrame
df_all = pd.concat(dataframes, ignore_index=True)
# List of all operators 
operators = list(files.keys())

def predict(df_op, model, model_name):
    # input values (Qual should be RSRQ)
    features = ["SNR", "Distance_km","Qual", "Speed"] 
    df_ml = df_op.dropna(subset=features + ["Level"])
    # input values
    X = df_ml[features]
    # output value
    y = df_ml["Level"]

    # dataset split
    # point where dataset is split
    split_idx = int(len(df_ml) * 0.9)
    # split input values
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    # split output values
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    # data training
    model.fit(X_train, y_train)
    # data prediction
    y_pred = model.predict(X_test)
    # print 
     

    df_ml = df_ml.copy()
    df_ml.loc[X_test.index, "Level_pred"] = y_pred

    return df_ml 
    


for model_name, model in models.items():

    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True, sharey=True)

    for ax, op in zip(axes, operators):

        df_op = df_all[df_all["Operator"] == op]
        df_pred = predict(df_op, model, model_name)

        ax.plot(
            df_pred["Distance_km"],
            df_pred["Level"],
            color=colors[op],
            alpha=0.4,
            label="Měřené"
        )

        ax.plot(
            df_pred["Distance_km"],
            df_pred["Level_pred"],
            color="black",
            linewidth=2,
            label="Predikce"
        )

        ax.set_title(op)
        ax.set_ylabel("RSRP [dBm]")
        ax.grid(True)
        ax.legend(loc='lower left')
        ax.xaxis.set_major_locator(MultipleLocator(50))

    axes[-1].set_xlabel("Vzdálenost [km]")

    #fig.suptitle(f"{model_name} – predikce RSRP", fontsize=18)
    fig.tight_layout()
    plt.savefig(f"{SaveFigLoc}/Metoda_{model_name}.pdf",format="pdf",bbox_inches="tight")
plt.show()









