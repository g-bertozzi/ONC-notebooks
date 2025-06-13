import pandas as pd
import onc
import os
import json
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np

from functools import reduce

# token = os.environ["TRICY_TOKEN"]
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("ONC_TOKEN")

# Create ONC client
my_onc = onc.ONC(token)

# NOTE: HARDCODED - global dataframe of sensor info
# schema: propertyCode, name, unit, deviceCategoryCode
sensor_info = pd.DataFrame([
    {"propertyCode": "oxygen", "name": "Dissolved Oxygen", "unit": "ml/l", "deviceCategoryCode": "OXYSENSOR"},
    {"propertyCode": "parphotonbased", "name": "PAR", "unit": "µmol/m²/s", "deviceCategoryCode": "radiometer"},
    {"propertyCode": "chlorophyll", "name": "Chlorophyll", "unit": "µg/l", "deviceCategoryCode": "FLNTU"},
    {"propertyCode": "seawatertemperature", "name": "Temperature", "unit": "°C", "deviceCategoryCode": "CTD"},
    {"propertyCode": "salinity", "name": "Salinity", "unit": "psu", "deviceCategoryCode": "CTD"},
    {"propertyCode": "turbidityntu", "name": "Turbidity", "unit": "NTU", "deviceCategoryCode": "FLNTU"},
])


def find_properties_by_location(locationCode: str):
    """
    Retrieves and prints a list of all sensor properties available at a given ONC location.
    For each property, includes its display name, property code, and whether it has associated device data.

    Parameters:
        locationCode (str): The ONC location code to query (e.g., "FGPPN").

    Returns:
        None: Prints a DataFrame of available properties to stdout.
    """
    params = {
        "locationCode": locationCode,
        #"deviceCategoryCode" : "CTD" 
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


def get_dataframe(start: str, end: str, props: list[str]) -> list[pd.DataFrame]:
    """
    Retrieves ONC scalar sensor data for a list of propertyCodes (e.g., 'oxygen', 'chlorophyll', etc.)
    over a specified time range at a fixed location (FGPPN). Each property is fetched from the appropriate
    deviceCategoryCode as defined in the global `sensor_info` table.

    Parameters:
        start (str): ISO 8601 formatted start time (e.g., "2021-07-15T12:00:00.000Z").
        end (str): ISO 8601 formatted end time (e.g., "2021-07-15T12:30:00.000Z").
        props (list[str]): A list of sensor property codes to fetch (e.g., ["oxygen", "chlorophyll"]).

    Returns:
        list[pd.DataFrame]: A list of individual DataFrames, each containing timestamped data
                            for one sensor property. Schema of each DataFrame: [timestamp, {property}]
    """
    dfs = []

    # iterate through each property wanted and access it's deviceCategory via global dataframe
    for prop in props: 
        device_cat = sensor_info.set_index("propertyCode").at[prop, "deviceCategoryCode"]
        
        # call helper to fetch data for each property
        df = get_property(start=start, end=end, locationCode="FGPPN", deviceCategoryCode=device_cat, propertyCode=prop)

        if df is not None:
            dfs.append(df)

    # NOTE: ERROR HANDLE if no data fetched
    if not dfs:
        print(f"No data returned for {props} from {start} to {end}")
        return
    
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="timestamp", how="outer"), dfs) # merge all the dataframes by joining on time
    merged_df.sort_values("timestamp", inplace=True) 

    # print(f"{props} returned {len(merged_df)} rows") # NOTE: debugging
    return merged_df

def get_property(start: str, end: str, locationCode: str, deviceCategoryCode: str, propertyCode: str = None) -> pd.DataFrame:
    """
    Fetches scalar data for specified single property from specified location and device category and time window. 
    Returns a  DataFrame with timestamps and sensor values.

    Parameters:
        start (str): Start date in ISO 8601 format (e.g., "2023-07-11T17:00:00.000Z").
        end (str): End date in ISO 8601 format (e.g., "2023-07-11T22:30:00.000Z").
        locationCode (str): ONC location code (e.g., "CF341").
        deviceCategoryCode (str): ONC device category (e.g., "CTD").
        propertyCode(str): Comma-separated sensor types to fetch (e.g., "depth,temperature").

    Returns:
        pd.DataFrame: DataFrame containing sensor values with a timestamp index.
                    schema: timestamp: datetime obj, {prop}: int
    """

    # TODO: fix oxygen to filter for sensorCategoryCode = "oxygen_corrected"
    #if propertyCode == "oxygen":

    # TODO: optimize - presently doing ruqest fro every porpertyCode, but I think I could limit this ß

    params = {
        "locationCode": locationCode,
        "deviceCategoryCode": deviceCategoryCode,
        "propertyCode": propertyCode,
        "dateFrom": start,
        "dateTo" : end,
        "metadata": "minimum",
        "qualityControl": "clean",
        "resamplePeriod": 1800,
        "resampleType": "avg"
        }
    
    # print(f"Requesting {propertyCode} from {start} to {end}") # NOTE: debugging
     
    # JSON response from ONC
    result = my_onc.getScalardata(params)

    # NOTE: ERROR HANDLE if there is no data returned
    if not result or "sensorData" not in result or result["sensorData"] is None or len(result["sensorData"]) == 0:
        print(f"No data returned for devices in {deviceCategoryCode} at {locationCode} between {start} and {end}.")
        return
        
    else:
        sensor = result["sensorData"][0] # isolate sensor
        
        # extract sensors data fields
        prop = sensor["propertyCode"]
        times = sensor["data"]["sampleTimes"]
        values = sensor["data"]["values"]

        # populate dataframe
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(times), # convert strings to datetime objects
            prop: values
        })

        # print(f"{prop} min time: {df["timestamp"].min()}") # NOTE: debugging
        # print(f"{prop} max time: {df["timestamp"].max()}") # NOTE: debugging

    df.sort_values("timestamp", inplace=True)
    
    #print(df.head()) # NOTE: debugging
    return df

def smooth_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies rolling mean smoothing and rolling z-score outlier filtering 
    to all data (i.e. numeric) columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'timestamp' and sensor data.

    Returns:
        pd.DataFrame: Smoothed and filtered DataFrame (same shape).
    """
    window = 30
    z_thresh = 3.0

    smoothed_df = df.copy()
    numeric_cols = smoothed_df.select_dtypes(include='number').columns.tolist()
    if 'timestamp' in numeric_cols:
        numeric_cols.remove('timestamp')

    for col in numeric_cols:
        roll_mean = smoothed_df[col].rolling(window=window, center=True).mean()
        roll_std = smoothed_df[col].rolling(window=window, center=True).std()
        z_scores = (smoothed_df[col] - roll_mean) / roll_std

        # Apply filter: keep rolling mean where z-score is within threshold
        smoothed_df[col] = roll_mean.where(z_scores.abs() < z_thresh, np.nan)

    # Interpolate missing values (or optionally: .dropna())
    smoothed_df[numeric_cols] = smoothed_df[numeric_cols].interpolate()

    return smoothed_df

def rename_columns_with_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames sensor data columns in a DataFrame by appending their display name and unit of measure,
    using the global `sensor_info` table for lookup. The 'timestamp' column is left unchanged.

    Parameters:
        df (pd.DataFrame): A DataFrame containing time-series sensor data. Assumes columns include
                           propertyCodes such as 'oxygen', 'salinity', etc.

    Returns:
        pd.DataFrame: A new DataFrame with sensor columns renamed to include human-readable names and units.
                      Example: 'oxygen' -> 'Dissolved Oxygen (ml/l)'
    """
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

def plot_all_sensors(df: pd.DataFrame, title: str = "Sensor Readings Over Time", ymax: float = None, ytick_freq: float = None) -> None:
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

    renamed_df = rename_columns_with_units(df) # rename columns with units

    # TODO: move this
    plot_df = smooth_df(renamed_df) # sub sample and filter for outliters

     # Set 'timestamp' as index
    plot_df = plot_df.set_index("timestamp")


    fig, ax = plt.subplots(figsize=(14, 10))

    # Get sensor columns (exclude timestamp)
    sensor_cols = plot_df.columns.tolist()

    for col in sensor_cols:
        # ax.plot(plot_df.index, plot_df[col], label=col, linewidth=0.8) # NOTE: plot without priority

        # NOTE: HARDCODED - plot with priority 
        base = col.lower().split(" (")[0] # Get base property name (remove units in parentheses)

        # Plot in order of importance
        if "oxygen" in base:
            ax.plot(plot_df.index, plot_df[col], label=col, linewidth=1, zorder=5)
        elif "par" in base:
            ax.plot(plot_df.index, plot_df[col], label=col, linewidth=1, zorder=4)
        elif "turbidity" in base:
            ax.plot(plot_df.index, plot_df[col], label=col, linewidth=1, zorder=3)
        else:
            ax.plot(plot_df.index, plot_df[col], label=col, linewidth=1, zorder=1)

    start_time = df["timestamp"].iloc[0]
    end_time = df["timestamp"].iloc[-1]

    # Axis settings
    ax.set_xlabel("Time")
    ax.set_ylabel("Sensor Value")
    ax.set_title(f"{title}\n"
                 f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}"
                 )

    # axis limits
    ax.set_ylim(bottom=0)  # Always start at 0
    if ymax:
        ax.set_ylim(top=ymax)
    #ax.set_xlim(left=start_time, right=end_time)

    if ytick_freq is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick_freq))
    
    # Format x-axis ticks as: "Jul 10, 2021 13:45:00"
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y %H:%M:%S'))
    # Rotate labels to avoid overlap
    fig.autofmt_xdate()

    # Grid and legend
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    ax.legend(title="Sensors", loc="upper right")

    plt.tight_layout()
    plt.show()

def subplot_each_sensor_with_oxygen(df: pd.DataFrame, title_prefix: str = "Oxygen vs", ytick_freq: float = None, oxygen_ytick_freq: float = None) -> None:
    """
    Creates a series of subplots where each subplot shows oxygen vs another sensor over time.

    Parameters:
        df (pd.DataFrame): DataFrame with 'timestamp' and multiple sensor columns.
        title_prefix (str): Prefix for each subplot's title.
        ytick_freq (float): Frequency of y-ticks on the right axis (sensor).
        oxygen_ytick_freq (float): Frequency of y-ticks on the left axis (oxygen).

    Returns:
        None
    """

    renamed_df = rename_columns_with_units(df) # rename columns with units

    # TODO: move this
    plot_df = smooth_df(renamed_df) # sub sample and filter for outliters

    # Set 'timestamp' as index
    plot_df = plot_df.set_index("timestamp")

    # Identify the oxygen column
    oxygen_col = [col for col in plot_df.columns if "oxygen" in col.lower()]
    if not oxygen_col:
        raise ValueError("No column containing 'oxygen' found.")
    oxygen_col = oxygen_col[0]

    # Get list of all other sensor columns
    sensor_cols = [col for col in plot_df.columns if col != oxygen_col]

    # Create subplots
    num_sensors = len(sensor_cols)
    fig, axs = plt.subplots(num_sensors, 1, figsize=(12, 3 * num_sensors), sharex=True)

    # Ensure axs is iterable
    if num_sensors == 1:
        axs = [axs]

    # Plot each sensor vs oxygen
    for i, sensor_col in enumerate(sensor_cols):
        ax = axs[i]
        ax2 = ax.twinx()

        ax.plot(plot_df.index, plot_df[oxygen_col], color='blue', label='Oxygen', linewidth=1)
        ax.set_ylabel('Oxygen', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        ax2.plot(plot_df.index, plot_df[sensor_col], color='red', label=sensor_col, linewidth=1)
        ax2.set_ylabel(sensor_col, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax.set_title(f"Oxygen vs {sensor_col}")
        ax.grid(True, linestyle='--', linewidth=0.5)

    # Shared x-axis label
    axs[-1].set_xlabel("Timestamp")

    # Format x-axis ticks as: "Jul 10, 2021 13:45:00"
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y %H:%M:%S'))
    fig.autofmt_xdate()

    start_time = df["timestamp"].iloc[0]
    end_time = df["timestamp"].iloc[-1]

    # Add overall title
    fig.suptitle(f"Folger Pinnacle\n{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}")

    # Adjust layout to make space for suptitle
    plt.subplots_adjust(top=0.93, hspace=0.4)

    plt.show()