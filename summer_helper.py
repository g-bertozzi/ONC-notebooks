"""
Simple plotting helpers for ONC summer notebooks.
Wraps ONC_client and onc_plotting for quick one-off calls.
"""

from datetime import datetime
from ONC_client import ONCSensorClient
from onc_plotting import subplot_all_with_oxygen, plot_dataframe_plotly, PLACE


def fetch_and_plot_summer_data(token: str, location_code: str = "FGPPN", plot_type: str = "oxygen") -> None:
    """
    Fetch 2025 summer data (Jan 1 to today) and plot it immediately.
    
    Args:
        token (str): ONC API token
        location_code (str): ONC location code (default "FGPPN")
        plot_type (str): "oxygen" for oxygen subplots, "plotly" for interactive chart
    """
    client = ONCSensorClient(token=token)
    
    # Time frame - 2025-01-01 through today
    start = "2025-01-01T00:00:00.000Z"
    today = datetime.utcnow().replace(microsecond=0).isoformat() + ".000Z"
    
    # Folger Pinnacle properties
    properties = ["oxygen", "salinity", "seawatertemperature", "sigmat", "chlorophyll", "turbidityntu"]
    
    # Resample to 30 min intervals
    resample = 1800
    
    # Fetch
    df = client.get_multi_property_dataframe(
        start=start,
        end=today,
        location_code=location_code,
        property_codes=properties,
        resample=resample
    )
    
    if df is None or df.empty:
        print(f"No data for {PLACE.get(location_code, {}).get('name', location_code)}")
        return
    
    # Plot
    if plot_type == "oxygen":
        subplot_all_with_oxygen(df=df, location_code=location_code)
    elif plot_type == "plotly":
        plot_dataframe_plotly(df=df, location_code=location_code)
    else:
        raise ValueError(f"Unknown plot_type '{plot_type}'. Choose: oxygen, plotly")
