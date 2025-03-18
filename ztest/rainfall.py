import requests
import json

# Define API endpoint
url = "https://power.larc.nasa.gov/api/temporal/monthly/point"

# Define parameters for location (India - Example: New Delhi)
params = {
    "latitude": 28.6139,  # Change this to your location
    "longitude": 77.2090,
    "parameters": "PRECTOT",  # Total Precipitation
    "start": 201201,  # Jan 2012
    "end": 202312,    # Dec 2023
    "community": "RE",
    "format": "JSON",
}

# Request data
response = requests.get(url, params=params)

# Print the raw response (for debugging)
print("Status Code:", response.status_code)
print("Response JSON:", response.text)

# Try to parse the JSON
try:
    data = response.json()
    rainfall = data["properties"]["parameter"]["PRECTOT"]
    print("Rainfall Data:", rainfall)
except json.JSONDecodeError:
    print("Error: Could not decode JSON.")
except KeyError as e:
    print(f"KeyError: {e}. The API response structure might have changed.")
