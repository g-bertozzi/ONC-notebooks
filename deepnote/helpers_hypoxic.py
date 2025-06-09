import pandas as pd
import onc
import os
import json
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import numpy as np

from functools import reduce

#token = os.environ["TRICY_TOKEN"]
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("ONC_TOKEN")

# Create ONC client
my_onc = onc.ONC(token)

# HARDCODED: global dataframe of sensor info
# schema: propertyCode, name, unit, deviceCategoryCode
sensor_info = pd.DataFrame([
    {"propertyCode": "oxygen", "name": "Dissolved Oxygen", "unit": "ml/l", "deviceCategoryCode": "OXYSENSOR", "priority"},
    {"propertyCode": "parphotonbased", "name": "PAR (Photon Based)", "unit": "µmol/m²/s", "deviceCategoryCode": "radiometer"},
    {"propertyCode": "chlorophyll", "name": "Chlorophyll", "unit": "µg/l", "deviceCategoryCode": "FLNTU"},
    {"propertyCode": "seawatertemperature", "name": "Temperature", "unit": "°C", "deviceCategoryCode": "CTD"},
    {"propertyCode": "salinity", "name": "Salinity", "unit": "psu", "deviceCategoryCode": "CTD"},
    {"propertyCode": "turbidityntu", "name": "Turbidity", "unit": "NTU", "deviceCategoryCode": "FLNTU"},
])

def find_properties_by_location(locationCode: str):
    """
    Parameters:
    
    Returns:
        None
    """
    params = {
        "locationCode": locationCode,
        #"deviceCategoryCode" : "CTD" # only consider CTD data properties
    }

    result = my_onc.getProperties(params)
    extracted = []

    for entry in result:
        # Defensive check: make sure these keys exist
        name = entry.get("propertyName", "")
        code = entry.get("propertyCode", "")
        has_data = entry.get("hasDeviceData", False)

        # Optionally: filter out properties that aren't actually measured
        if name and code:
            extracted.append({
                "propertyName": name,
                "propertyCode": code,
                "hasDeviceData": has_data
            })
    
    df = pd.DataFrame(extracted)
    print(df)

def get_property(start: str, end: str, locationCode: str, deviceCategoryCode: str, propertyCode: str) -> pd.DataFrame:
    """
    Fetches scalar data using the ONC Python SDK for a given location, device category, sensor propert(ies),
    and time window. Returns a merged DataFrame with timestamps, sensor values, and units of measurement.

    Parameters:
        start (str): Start date in ISO 8601 format (e.g., "2023-07-11T17:00:00.000Z").
        end (str): End date in ISO 8601 format (e.g., "2023-07-11T22:30:00.000Z").
        locationCode (str): ONC location code (e.g., "CF341").
        deviceCategoryCode (str): ONC device category (e.g., "CTD").
        propertyCode(str): Comma-separated sensor types to fetch (e.g., "depth,temperature").

    Returns:
        pd.DataFrame: DataFrame containing merged sensor values with a timestamp index.
                    schema: timestamp: datetime obj, {prop}: int
    """

    params = {
        "locationCode": locationCode,
        "deviceCategoryCode": deviceCategoryCode,
        "propertyCode": propertyCode,
        "dateFrom": start,
        "dateTo" : end
    }

    # JSON response from ONC
    result = my_onc.getScalardata(params)

    # error handle if there is no data returned
    if not result or "sensorData" not in result or result["sensorData"] is None or len(result["sensorData"]) == 0:
        print(f"No data returned for devices in {deviceCategoryCode} at {locationCode} between {start} and {end}.")
        return
        
    else:
        # extract the relevant sensors from the JSON response
        dfs = []

        for sensor in result["sensorData"]:
            # extract each sensors data fields
            prop = sensor["propertyCode"]
            times = sensor["data"]["sampleTimes"]
            values = sensor["data"]["values"]
            #unit = sensor["unitOfMeasure"]

            # populate dataframe
            df = pd.DataFrame({
                # syntax: "label": variable
                "timestamp": pd.to_datetime(times), # convert strings to datetime objects
                prop: values
            })
            dfs.append(df)

    # merge dataframes by joining on timestamp    
    df_merged = reduce(lambda left, right: pd.merge(left, right, on="timestamp", how="outer"), dfs)
    df_merged.sort_values("timestamp", inplace=True)
    
    df_merged.head()
    return df_merged

def get_dataframe(start: str, end: str, props: list[str]) -> list[pd.DataFrame]:
    """ 
    Parameters:
        props: 

    Returns:
        pd.DataFrame: Dataframe of merged dataframes
                schema: ex. timestamp, salinity, seawatertemperature, oxygen, parphotonbased, chlorophyll, turbidityntu
    """
    dfs = []

    for prop in props:
        device_cat = sensor_info.set_index("propertyCode").at[prop, "deviceCategoryCode"]
        df = get_property(start=start, end=end, locationCode="FGPPN", deviceCategoryCode=device_cat, propertyCode=prop)
        if df is not None:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="timestamp", how="outer"), dfs)
    merged_df.sort_values("timestamp", inplace=True)

    return merged_df


def rename_columns_with_units(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}

    # iterate through each column
    for col in df.columns:
        if col == "timestamp":
            continue
        
        # search for current col
        match = sensor_info[sensor_info["propertyCode"] == col]
        if not match.empty:
            name = match.iloc[0]["name"]
            unit = match.iloc[0]["unit"]
            renamed[col] = f"{name} ({unit})"

    return df.rename(columns=renamed)

def plot_all_sensors(df: pd.DataFrame, title: str = "Sensor Readings Over Time", ymax: float = None) -> None:
    """
    Plots each numeric sensor column in the DataFrame against time, 
    with line priority and unit-labeled legend entries.

    Parameters:
        df (pd.DataFrame): DataFrame with 'timestamp' and sensor columns.
        title (str): Title of the plot.
        ymax (float): Optional maximum y-axis value. Default shows all values.

    Returns:
        None
    """
    # Copy to avoid modifying original DataFrame
    plot_df = rename_columns_with_units(df)

    # Convert timestamp if needed and set as index
    if not pd.api.types.is_datetime64_any_dtype(plot_df["timestamp"]):
        plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])
    plot_df = plot_df.set_index("timestamp")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Get sensor columns (exclude timestamp)
    sensor_cols = plot_df.columns.tolist()

    for col in sensor_cols:
        # ax.plot(plot_df.index, plot_df[col], label=col, linewidth=0.8)

        # HARDCODED: plot with priority 
        # Get base property name (remove units in parentheses)
        base = col.lower().split(" (")[0]

        # Plot in order of importance
        if "oxygen" in base:
            ax.plot(plot_df.index, plot_df[col], label=col, linewidth=0.8, zorder=5)
        elif "par" in base:
            ax.plot(plot_df.index, plot_df[col], label=col, linewidth=0.8, zorder=4)
        elif "turbidity" in base:
            ax.plot(plot_df.index, plot_df[col], label=col, linewidth=0.8, zorder=3)
        else:
            ax.plot(plot_df.index, plot_df[col], label=col, linewidth=0.8, zorder=1)

    # Axis settings
    ax.set_xlabel("Time")
    ax.set_ylabel("Sensor Value")
    ax.set_title(title)

    # Y-axis limits
    ax.set_ylim(bottom=0)  # Always start at 0
    if ymax:
        ax.set_ylim(top=ymax)

    # Grid and legend
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    ax.legend(title="Sensors", loc="upper right")

    plt.tight_layout()
    plt.show()
