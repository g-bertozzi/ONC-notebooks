"""
ONC (Ocean Networks Canada) API client for fetching sensor data.
"""

from functools import reduce   # Library used to combine DataFrames
import pandas as pd            # Library used for data analysis; referred to as 'pd'
import onc                     # ONC's client library

# Properties already defined for this code include: oxygen, par, chlorophyll, seawatertemperature, salinity, turbidity, density, sigmat
# NOTE: if you would like to add additional properties use this schema: property_code: {label, deviceCategoryCode}
SENSOR_INFO: dict[str:str] = {
    "oxygen": {
        "label": "Oxygen (ml/l)",
        "deviceCategoryCode": "OXYSENSOR",
    }, "parphotonbased": {
        "label": "PAR (µmol/m²/s)",
        "deviceCategoryCode": "radiometer",
    }, "chlorophyll": {
        "label": "Chlorophyll (µg/l)",
        "deviceCategoryCode": "FLNTU",
    }, "seawatertemperature": {
        "label": "Temperature (°C)",
        "deviceCategoryCode": "CTD",
    }, "salinity": {
        "label": "Salinity (psu)",
        "deviceCategoryCode": "CTD",
    }, "turbidityntu": {
        "label": "Turbidity (NTU)",
        "deviceCategoryCode": "FLNTU",
    }, "density": {
        "label": "Density (kg/m3)",
        "deviceCategoryCode": "CTD",
    }, "sigmat": {
        "label": "Sigma-t (kg/m3)",
        "deviceCategoryCode": "CTD",
    }
}

class ONCSensorClient:
    """
    A client for fetching property data from the ONC API using the onc Client library.
    Initializes with a user-provided token and provides methods for making getScalardata requests 
    from the API per property_code, processing each API response, and merging all data into a single DataFrame.
    """
    # TODO: comment on the constructor, methods, properties/getScalardata specificity
    def __init__(self, token: str) -> None:
        """
        Initializes the ONCSensorClient with a user-provided ONC API token.

        Inputs:
            token (str): ONC API token for authentication
        """
        self.onc = onc.ONC(token=token)

    # ========== API CALL METHODS ==========

    def fetch_property_result(self, start: str, end: str, location_code: str, property_code: str, updates: bool = False, resample: int = None) -> dict:
        """
        Requests data from the ONC API using the getScalardata method from the onc Client library.
        Handles getScalardata call for a single property_code.

        Inputs:
            start (str): ISO 8601 UTC start timestamp (e.g., "2021-07-01T00:00:00Z")
            end (str): ISO 8601 UTC end timestamp (e.g., "2021-07-02T00:00:00Z")
            location_code (str): ONC location code (e.g., "FGPPN")
            property_code (str): ONC property code (e.g., "oxygen", "seawatertemperature")
            updates (bool, optional): If True, prints API call details for debugging. Defaults to False.
            resample (int, optional): Resample period in seconds (e.g., 60 for 1-minute averages). Defaults to None.

        Returns:
            dict: JSON-like dictionary from ONC API containing the requested scalar data
        """
        device_cat = SENSOR_INFO[property_code]["deviceCategoryCode"]

        # Default parameters
        params = {
            "locationCode": location_code,
            "deviceCategoryCode": device_cat,
            "propertyCode": property_code,
            "dateFrom": start,
            "dateTo": end,
        }

        # If resampling
        if resample:
            params["metadata"] = "minimum"
            params["qualityControl"] = "clean"
            params["resamplePeriod"] = resample
            params["resampleType"] = "avg"

        # For oxygen: multiple types of oxygen available
        if device_cat == "OXYSENSOR":
            params["sensorCategoryCodes"] = "oxygen_corrected"

            # For oxygen at FGPD: multiple oxygen sensors available
            if location_code == "FGPD":
                params["locationCode"] = "FGPD.O2"

        if updates:
            print(f"API Request: getScalardata{params}")

        result = self.onc.getScalardata(params)
        return result

    def result_to_dataframe(self, result: dict, property_code: str) -> pd.DataFrame:
        """
        Converts API getScalardata response for a single property_code to a labeled, time-indexed DataFrame.

        Inputs:
            result (dict): JSON-like response from ONC API for a single property_code.
                           Expected to contain 'sensorData' with 'sampleTimes' and 'values'.
            property_code (str): The ONC property_code used to fetch the data (e.g., "oxygen").

        Returns:
            pd.DataFrame: A DataFrame indexed by timestamp with a single column for the specified property.
                          Returns None if no data is found.
        """
        # Error handle if no API response
        if not result or "sensorData" not in result or not result["sensorData"]:
            print(f"No data for {property_code}")
            return None

        # Extract info from API response
        sensor = result["sensorData"][0]
        times = sensor["data"]["sampleTimes"]
        values = sensor["data"]["values"]
        column_title = SENSOR_INFO[property_code]["label"] if property_code in SENSOR_INFO else property_code

        # Create new DataFrame
        df = pd.DataFrame({
            "Time": pd.to_datetime(times),
            column_title: values
        })

        # Set timestamps as the index of the DataFrame; sort the DataFrame by time
        df.set_index("Time", inplace=True)
        df.sort_index(inplace=True)
        return df

    def get_multi_property_dataframe(self, start: str, end: str, location_code: str, property_codes: list[str], resample: int = None, updates: bool = False) -> pd.DataFrame:
        """
        Fetches, formats, and merges multiple properties into one time-indexed DataFrame.

        Inputs:
            start (str): Start datetime in ISO 8601 format (e.g., "2023-07-11T00:00:00.000Z").
            end (str): End datetime in ISO 8601 format.
            location_code (str): ONC location code (e.g., "FGPPN").
            property_codes (list[str]): List of ONC property_codes to fetch (e.g., ["oxygen", "seawatertemperature"]).
            resample (int, optional): Resample period in seconds. If set, API will average data over this period. Defaults to None.
            updates (bool, optional): If True, prints status updates during execution. Defaults to False.

        Returns:
            pd.DataFrame: A time-indexed DataFrame with one column per property.
                          Columns are labeled according to the `SENSOR_INFO` dictionary.
                          Returns None if no data is retrieved.
        """
        dfs = [] # List to hold DataFrame for each property

        # Make API request and DataFrame for each property
        for prop in property_codes:
            try:
                result = self.fetch_property_result(start, end, location_code, prop, updates, resample)
                df = self.result_to_dataframe(result, prop)
                if df is not None:
                    dfs.append(df) # Append DataFrame to list
            except Exception as e: # Print any errors
                print(f"Error retrieving {prop}: {e}")

        if not dfs: # If no data was retrieved for any property return nothing
            return None

        # Merge each property DataFrame into one single DataFrame for all properties
        merged_df = reduce(lambda left, right: pd.merge(left, right, on="Time", how="outer"), dfs)
        merged_df.sort_index(inplace=True)

        if updates:
            print(f"Generated combined data frame for {property_codes} at {location_code}.")

        return merged_df
