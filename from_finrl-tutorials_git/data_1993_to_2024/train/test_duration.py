import pandas as pd
from datetime import datetime as dt, timedelta

def calculate_duration(start: dt, end: dt):
    # Calculate the duration
    duration = end - start

    # Extract total seconds from duration
    total_seconds = int(duration.total_seconds())
    hours, remainder = divmod(abs(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    # Create the dataframe with the required format
    data = {
        "start": [start.strftime("%Y-%m-%d_%H:%M:%S")],
        "end": [end.strftime("%Y-%m-%d_%H:%M:%S")],
        "duration_hour": [hours],
        "duration_minute": [minutes],
        "duration_second": [seconds],
    }
    return pd.DataFrame(data)

# Test the function with the provided example
start = dt.now()
end = start + timedelta(seconds=60)  # Simulate a duration of 1 minute
df = calculate_duration(start, end)
print(df)
