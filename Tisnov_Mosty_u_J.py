# Required imports are insert here
import numpy as np
import pandas as pd
import tkinter as tk
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

from pathlib import Path
from pyproj import Transformer
from matplotlib.ticker import FuncFormatter, MultipleLocator


# Type here location of your input files
files = {
    "T-Mobile (1)": "dataset_Tisnov_Mosty\\T_Mobile1.csv",
    "T-Mobile (2)": "dataset_Tisnov_Mosty\\T_Mobile2.csv",
    "T-Mobile (3)": "dataset_Tisnov_Mosty\\T_Mobile3.csv"
}
# Color settings for operators
colors = {"T-Mobile (1)": "green","T-Mobile (2)": "orange","T-Mobile (3)": "purple"}

# Here define lacation whrere you want to save figures as PDF
SaveFigLoc = "Tisnov_results"

# Limit values
LevelMaxRSRP = -50
LevelMinRSRP = -130

# setting the size of the figures
FigSizeSet = (26,14)

# Setting the number of bins in histogram
SNRbins = 20
RSRPbins = 20

# Setting fugure text style to LaTex font and font size
unitFontSize = 20
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "axes.titlepad": 24,
"axes.labelsize": unitFontSize, "axes.titlesize": unitFontSize, "xtick.labelsize": unitFontSize, "ytick.labelsize": unitFontSize, "legend.fontsize": unitFontSize, "figure.titlesize": unitFontSize})

# listing of band combinations
BAND_MHz = {
"N28": "700 MHz",
"N20": "800 MHz",
"N8": "900 MHz",
"N3": "1800 MHz",
"N1": "2100 MHz",
"N38": "2600 MHz",
"N7": "2600 MHz",
"N78": "3500 MHz",
"N48": "3600 MHz",

"L1": "2100 MHz",
"L3": "1800 MHz",
"L7": "2600 MHz",
"L8": "900 MHz",
"L20": "800 MHz",
"L38": "2600 MHz",
"L28": "700 MHz",

"EGSM": "900 MHz",
"PGSM": "900 MHz",
}

# GSM frequency bands (2G)
GSM_BANDS = {"EGSM", "PGSM"}  
# LTE frequency bands (4G)                   
LTE_BANDS = {"L1","L3","L7","L8","L20","L28","L38"} 
# NR frequency bands (5G)
NR_BANDS  = {"N1","N3","N7","N8","N28","N38","N78"} 


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
    df['dist'] = pd.Series(dist)
    # delete error values
    df = df[df['dist'] < 80]  
    # distance conversion to kilometers
    df['Distance_km'] = pd.Series(df["dist"]).cumsum() / 1000
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


## GPS unit conversion from meters to degrees (for asis of map figures)
transformer = Transformer.from_crs(3857, 4326, always_xy=True)

def format_lon(x, pos):
    lon, lat = transformer.transform(x, 0)
    return f"{lon:.2f}°"

def format_lat(y, pos):
    lon, lat = transformer.transform(0, y)
    return f"{lat:.2f}°"


""" ##  Display of RSRP parameter development over time
# for each date a subplot of 3 operators and a comparative is drawn
fig, axes = plt.subplots(4, 1, figsize=FigSizeSet, sharex=True, sharey=True)
axes = axes.flatten()

# subplots of 3 operators 
for i, op in enumerate(operators):
    df_op = df_all[df_all["Operator"] == op]
    axes[i].plot(df_op['datetime'], df_op['Level'], color=colors[op], linewidth=1)
    axes[i].set_ylabel("RSRP [dBm]")
    axes[i].grid(True)
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

# comparative subplot
ax_all = axes[3]
for op in operators:
    df_op = df_all[df_all["Operator"] == op]
    ax_all.plot(df_op['datetime'], df_op['Level'], label=op, color=colors[op], linewidth=1)

#axes[0].set_title(f"RSRP v závislosti na čase - {day}")
ax_all.set_xlabel("Čas [hh:mm:ss]")
ax_all.set_ylabel("RSRP [dBm]")
ax_all.legend()
ax_all.grid(True)
fig.tight_layout()

# save figure in pdf format
#plt.savefig(f"{SaveFigLoc}/RSRPtime_{day}.pdf", bbox_inches="tight")
 #-------------------------------------------------------------------------------------------------------------------- """
 

##  Display of RSRP parameter development over distance
fig, axes = plt.subplots(4, 1, figsize=FigSizeSet, sharex=True, sharey=True)
axes = axes.flatten()

# subplots of 3 operators
for i, op in enumerate(operators):
    df_op = df_all[df_all["Operator"] == op]
    axes[i].plot(df_op['Distance_km'], df_op['Level'], color=colors[op])
    axes[i].set_ylabel("RSRP [dBm]")
    axes[i].grid(True)

# comparative subplot
ax_all = axes[3]
for op in operators:
    df_op = df_all[df_all["Operator"] == op]
    ax_all.plot(df_op['Distance_km'], df_op['Level'], label=op, color=colors[op])

#axes[0].set_title("RSRP v závislosti na vzdálenosti")
ax_all.set_xlabel("vzdálenost [km]")
ax_all.set_ylabel("RSRP [dBm]")
ax_all.legend()
ax_all.grid(True)
# setting the x axis numbering to 50 km per division
ax_all.xaxis.set_major_locator(MultipleLocator(50))
fig.tight_layout()

# save figure in pdf format
plt.savefig(f"{SaveFigLoc}/RSRPdistance.pdf", bbox_inches="tight")
#--------------------------------------------------------------------------------------------------------------------


##  Display HISTOGRAMS of parameter RSRP
fig, axes = plt.subplots(4, 1, figsize=FigSizeSet, sharex=True)
axes = axes.flatten()

# subplots of 3 operators
for i, op in enumerate(operators):
    df_op = df_all[df_all["Operator"] == op]
    axes[i].hist(df_op["Level"].dropna(), bins=RSRPbins, color=colors[op])
    axes[i].set_ylabel("Četnost")
    axes[i].grid(True)
    
# comparative subplot
ax_all = axes[3]
for op in operators:
    ax_all.hist(df_all[df_all["Operator"] == op]["Level"].dropna(), bins=RSRPbins, histtype='step', linewidth=1.5, color=colors[op], label=op)
    
#axes[0].set_title("Histogram parametru RSRP")   
ax_all.set_xlabel("RSRP [dBm]")
ax_all.set_ylabel("Četnost")
ax_all.legend()
ax_all.grid(True)
fig.tight_layout()

# save figure in pdf format
plt.savefig(f"{SaveFigLoc}/HistRSRP.pdf", bbox_inches="tight")
#--------------------------------------------------------------------------------------------------------------------


##  Display BOXPLOTS of parameter RSRP
fig, ax = plt.subplots(figsize=FigSizeSet)

data_box = [df_all[df_all["Operator"] == op]["Level"].dropna() for op in operators]
bp = ax.boxplot(data_box, tick_labels=operators, patch_artist=True)

for patch, op in zip(bp['boxes'], operators):
    patch.set_facecolor('none')
    patch.set_edgecolor(colors[op])
    patch.set_linewidth(1.8)
#ax.set_title("Boxplot parametru RSRP")    
ax.set_ylabel("RSRP [dBm]")
ax.grid(True)

# save figure in pdf format
plt.savefig(f"{SaveFigLoc}/BoxRSRP.pdf", bbox_inches="tight")
#--------------------------------------------------------------------------------------------------------------------


##  Display HISTOGRAMS of parameter SNR
fig, ax = plt.subplots(figsize=FigSizeSet)

data_box_snr = [df_all[df_all["Operator"] == op]["SNR"].dropna() for op in operators]
bp = ax.boxplot(data_box_snr, tick_labels=operators, patch_artist=True)

for patch, op in zip(bp['boxes'], operators):
    patch.set_facecolor('none')
    patch.set_edgecolor(colors[op])
    patch.set_linewidth(1.8)
#ax.set_title("Histogram parametru SNR")    
ax.set_ylabel("SNR [dB]")
ax.grid(True)

# save figure in pdf format
plt.savefig(f"{SaveFigLoc}/BoxSNR.pdf", bbox_inches="tight")
#--------------------------------------------------------------------------------------------------------------------


## RSRP depending on location supported by map
for op, df_op in df_all.groupby("Operator"):
    fig, ax = plt.subplots(figsize=FigSizeSet)
    # get our gps location and transfer Longitude and Latitude to Web Mercator (meters) - basemap require this
    gdf = gpd.GeoDataFrame(df_op, geometry=gpd.points_from_xy(df_op["Longitude"], df_op["Latitude"]), crs="EPSG:4326").to_crs(epsg=3857)
    # plot RSRP on location
    gdf.plot(ax=ax, column="Level", cmap="viridis", markersize=50, legend=True, vmin=df_all["Level"].min(), vmax=df_all["Level"].max(), legend_kwds={"label": "RSRP [dBm]"})
    # add map under trace
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    # Set equal scaling for both x and y axes
    ax.set_aspect('equal')
    # Show Longitude and Latitude in degrees
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    #ax.set_title(f"RSRP v závislosti na pozici – {op}")
    ax.set_xlabel("Zeměpisná délka [°]")
    ax.set_ylabel("Zeměpisná šířka [°]")

    # save figure in pdf format
    plt.savefig(f"{SaveFigLoc}/RSRPmap_{op}.pdf", format="pdf", bbox_inches="tight")
#--------------------------------------------------------------------------------------------------------------------


## SNR depending on location supported by map
for op, df_op in df_all.groupby("Operator"):
    fig, ax = plt.subplots(figsize=FigSizeSet)
    # get our gps location and transfer Longitude and Latitude to Web Mercator (meters) - basemap require this
    gdf = gpd.GeoDataFrame(df_op, geometry=gpd.points_from_xy(df_op["Longitude"], df_op["Latitude"]), crs="EPSG:4326").to_crs(epsg=3857)
    # plot SNR on location
    gdf.plot(ax=ax, column="SNR", cmap="viridis", markersize=50, legend=True, vmin=df_all["SNR"].min(), vmax=df_all["SNR"].max(), legend_kwds={"label": "SNR [dB]"} )
    # add map under trace
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    # Set equal scaling for both x and y axes
    ax.set_aspect('equal')
    # Show Longitude and Latitude in degrees
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    #ax.set_title(f"SNR v závislosti na pozici – {op}")
    ax.set_xlabel("Zeměpisná délka [°]")
    ax.set_ylabel("Zeměpisná šířka [°]")
    
    # save figure in pdf format
    #plt.savefig(f"{SaveFigLoc}/polohaSNR_{op}.pdf", format="pdf", bbox_inches="tight") 
#--------------------------------------------------------------------------------------------------------------------  
    
## Technology and band depending on location supported by map
for op, df_op in df_all.groupby("Operator"):
    fig, ax = plt.subplots(figsize=FigSizeSet)
    # find out not NaN rows in BAND colum
    df_op = df_op[df_op["BAND"].notna()].copy()
    # keep only valid combinations of radio access technology and RF band
                  # 2G – GSM bands only (EGSM, PGSM)
    df_op = df_op[((df_op["NetworkTech"] == "2G") & (df_op["BAND"].isin(GSM_BANDS))) |
                  # 4G – LTE bands only (L1, L3, L7, L8, L20, L28, L38)
                  ((df_op["NetworkTech"] == "4G") & (df_op["BAND"].isin(LTE_BANDS))) |
                  # 5G – NR bands only (N1, N3, N7, N8, N28, N38, N78)
                  ((df_op["NetworkTech"] == "5G") & (df_op["BAND"].isin(NR_BANDS)))
    ]
    # create technology band labels to legend
    df_op["TECH_BAND"] = (df_op["NetworkTech"].astype(str) + " " + df_op["BAND"].astype(str) + " (" + df_op["BAND"].map(BAND_MHz).fillna("?") + ")")
    # get our gps location and transfer Longitude and Latitude to Web Mercator (meters) - basemap require this
    gdf = gpd.GeoDataFrame(df_op, geometry=gpd.points_from_xy(df_op["Longitude"], df_op["Latitude"]), crs="EPSG:4326").to_crs(epsg=3857)
    # Create sorted category labels - give theme ID and do colormap  
    categories = sorted(gdf["TECH_BAND"].unique())
    gdf["cat_id"] = gdf["TECH_BAND"].astype("category").cat.codes
    cmap = plt.get_cmap("tab20").resampled(len(categories))
    # plot Technology and band on location
    gdf.plot(ax=ax, column="cat_id", cmap=cmap, markersize=25, categorical=True, legend=False)
     # add map under trace
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    # Set equal scaling for both x and y axes
    ax.set_aspect("equal")
    # Show Longitude and Latitude in degrees
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    #ax.set_title(f"Tachnologie a band v závislosti na pozici – {op}")
    ax.set_xlabel("Zeměpisná délka [°]")
    ax.set_ylabel("Zeměpisná šířka [°]")
    # Create legend handles
    handles = [mpatches.Patch(color=cmap(i), label=cat)
               for i, cat in enumerate(categories)]
    # set legend
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    
    # save figure in pdf format
    plt.savefig(f"{SaveFigLoc}/TechBand_{op}.pdf",format="pdf",bbox_inches="tight")
 #--------------------------------------------------------------------------------------------------------------------
    
    
plt.show()
