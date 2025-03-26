from flask import Flask, jsonify, send_from_directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime

app = Flask(__name__)

commodity_dict = {
    "arhar": "static/Arhar.csv",
    "bajra": "static/Bajra.csv",
    "barley": "static/Barley.csv",
    "copra": "static/Copra.csv",
    "cotton": "static/Cotton.csv",
    "sesamum": "static/Sesamum.csv",
    "gram": "static/Gram.csv",
    "groundnut": "static/Groundnut.csv",
    "jowar": "static/Jowar.csv",
    "maize": "static/Maize.csv",
    "masoor": "static/Masoor.csv",
    "moong": "static/Moong.csv",
    "niger": "static/Niger.csv",
    "paddy": "static/Paddy.csv",
    "ragi": "static/Ragi.csv",
    "rape": "static/Rape.csv",
    "jute": "static/Jute.csv",
    "safflower": "static/Safflower.csv",
    "soyabean": "static/Soyabean.csv",
    "sugarcane": "static/Sugarcane.csv",
    "sunflower": "static/Sunflower.csv",
    "urad": "static/Urad.csv",
    "wheat": "static/Wheat.csv"
}

annual_rainfall = [34,26,42,35,57,165,320,275,195,76,44,14,]
base = {
    "Paddy": 1245.5,
    "Arhar": 3200,
    "Bajra": 1175,
    "Barley": 980,
    "Copra": 5100,
    "Cotton": 3600,
    "Sesamum": 4200,
    "Gram": 2800,
    "Groundnut": 3700,
    "Jowar": 1520,
    "Maize": 1175,
    "Masoor": 2800,
    "Moong": 3500,
    "Niger": 3500,
    "Ragi": 1500,
    "Rape": 2500,
    "Jute": 1675,
    "Safflower": 2500,
    "Soyabean": 2200,
    "Sugarcane": 2250,
    "Sunflower": 3700,
    "Urad": 4300,
    "Wheat": 1350

}
commodity_list = []

class Commodity:
    def __init__(self, csv_name, base_price):
        self.name = os.path.basename(csv_name).split('.')[0]
        self.base_price = base_price
        
        # Load data with correct column names
        self.df = pd.read_csv(csv_name)
        
        # Validate required columns exist
        required_cols = ['Month', 'Year', 'Rainfall', 'WPI']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        # Feature engineering
        self._add_features()
        
        # Store feature names before conversion to numpy
        self.feature_columns = self.df.drop(columns=['WPI']).columns
        
        # Prepare features and target
        self.X = self.df.drop(columns=['WPI']).values
        self.y = self.df['WPI'].values
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(self.X_scaled, self.y)

    def _add_features(self):
        """Add time-based features"""
        # Add quarter (1-4)
        self.df['Quarter'] = (self.df['Month'] - 1) // 3 + 1
        
        # Add growth phase (1=sowing, 2=growing, 3=harvest)
        self.df['Growth_Phase'] = self.df['Month'].apply(
            lambda m: 1 if m in [6,7] else (2 if m in [8,9,10] else 3))
        
        # Add rainfall lags
        for lag in [1, 2, 3, 12]:
            self.df[f'Rainfall_Lag_{lag}'] = self.df['Rainfall'].shift(lag)
        
        # Drop rows with missing values
        self.df = self.df.dropna()

    def predict_wpi(self, month, year, rainfall):
        """Predict WPI using correct feature structure"""
        # Create input with same features as training
        input_data = {
            'Month': [month],
            'Year': [year],
            'Rainfall': [rainfall],
            'Quarter': [(month - 1) // 3 + 1],
            'Growth_Phase': [1 if month in [6,7] else (2 if month in [8,9,10] else 3)],
            'Rainfall_Lag_1': [self.df['Rainfall'].iloc[-1]],
            'Rainfall_Lag_2': [self.df['Rainfall'].iloc[-2] if len(self.df) > 1 else rainfall],
            'Rainfall_Lag_3': [self.df['Rainfall'].iloc[-3] if len(self.df) > 2 else rainfall],
            'Rainfall_Lag_12': [self.df['Rainfall'].iloc[-12] if len(self.df) > 11 else rainfall]
        }
        
        # Convert to DataFrame with correct column order
        input_df = pd.DataFrame(input_data)[self.feature_columns]
        
        # Scale and predict
        scaled_input = self.scaler.transform(input_df)
        return self.model.predict(scaled_input)[0]

# Load commodities with error handling
for name, path in commodity_dict.items():
    try:
        # Verify file exists before trying to load
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        base_price = base.get(name.capitalize())
        if base_price is None:
            print(f"No base price found for {name}")
            continue
            
        commodity = Commodity(path, base_price)
        commodity_list.append(commodity)
        print(f"Successfully initialized {name}")
        
    except Exception as e:
        print(f"Failed to load {name}: {str(e)}")
        continue
# for name, path in commodity_dict.items():
    try:
        print(path)
        base_price = base[name.capitalize()]
        commodity_list.append(Commodity(path, base_price))
        print(f"Loaded {name} successfully")
    except Exception as e:
        print(f"Error loading {name}: {str(e)}")


def get_price(wpi, commodity_name):
    """
    More concise version using dictionary comprehension
    """
    try:
        # Case-insensitive lookup in base prices
        base_price = next(
            v for k, v in base.items() 
            if k.lower() == commodity_name.lower()
        )
        return round((wpi / 100) * base_price, 2)
    except StopIteration:
        print(f"Warning: Commodity '{commodity_name}' not found in base prices")
        return None
    

# Helper Functions
def TwelveMonthsForecast(name):
    """Generate 12-month WPI forecast"""
    commodity = next((c for c in commodity_list if c.name.lower() == name.lower()), None)
    if not commodity:
        return None, None, []
    
    current_date = datetime.now()
    forecast = []
    
    for i in range(1, 13):
        month = (current_date.month + i - 1) % 12 or 12
        year = current_date.year + (current_date.month + i - 1) // 12
        rainfall = annual_rainfall[month-1]
        
        wpi = commodity.predict_wpi(month, year, rainfall)
        get_price_temp = get_price(wpi, name)
        # print("In 12 monthe forecast", get_price_temp)
        forecast.append((f"{month}/{year}", get_price_temp))
    
    if forecast:
        return max(v[1] for v in forecast), min(v[1] for v in forecast), forecast
    return None, None, []

def TwelveMonthPrevious(name):
    """Get previous 12 months of WPI data"""
    commodity = next((c for c in commodity_list if c.name.lower() == name.lower()), None)
    if not commodity:
        return []
    
    previous = []
    current_date = datetime.now()
    
    for i in range(12, 0, -1):
        month = (current_date.month - i) % 12 or 12
        year = current_date.year - (1 if (current_date.month - i) <= 0 else 0)
        
        # Try to find historical record
        mask = (commodity.df['Month'] == month) & (commodity.df['Year'] == year)
        if any(mask):
            wpi = commodity.df.loc[mask, 'WPI'].values[0]
            get_price_temp = get_price(wpi, name)
            # print("In 12 monthe previous", get_price_temp)
            previous.append((f"{month}/{year}", get_price_temp))
    
    return previous

def CurrentMonth(name):
    """Get current month price"""
    commodity = next((c for c in commodity_list if c.name.lower() == name.lower()), None)
    if not commodity:
        return None
    
    current_date = datetime.now()
    month = current_date.month
    year = current_date.year
    
    # Try to find existing data
    mask = (commodity.df['Month'] == month) & (commodity.df['Year'] == year)
    if any(mask):
        return commodity.df.loc[mask, 'WPI'].values[0]
    
    # If no data, predict
    avg_rainfall = annual_rainfall[month-1]
    get_temp_wpi = commodity.predict_wpi(month, year, avg_rainfall)
    print("in curreny month " , get_temp_wpi)
    return get_price(get_temp_wpi,name)
    print(name)
    # return commodity.predict_wpi(month, year, avg_rainfall)


# API Endpoint
@app.route('/commodity/<name>')
def crop_profile(name):
    max_crop, min_crop, forecast_crop_values = TwelveMonthsForecast(name)
    prev_crop_values = TwelveMonthPrevious(name)
    
    forecast_x = [i[0] for i in forecast_crop_values]
    forecast_y = [i[1] for i in forecast_crop_values]
    previous_x = [i[0] for i in prev_crop_values]
    previous_y = [i[1] for i in prev_crop_values]
    current_price = CurrentMonth(name)
    
    # Generate visualization
    image_url = generate_price_trend_chart(name, previous_x, previous_y, forecast_x, forecast_y)
    
    # Get additional crop data (you'll need to implement this)
    crop_data = get_crop_data(name)  # Replace with your data source
    
    context = {
        "name": name,
        "max_crop": max_crop,
        "min_crop": min_crop,
        "forecast_values": forecast_crop_values,
        "forecast_x": forecast_x,
        "forecast_y": forecast_y,
        "previous_values": prev_crop_values,
        "previous_x": previous_x,
        "previous_y": previous_y,
        "current_price": current_price,
        "vis_url": image_url, 
        "image_url": crop_data.get("image", ""),
        "prime_loc": crop_data.get("prime_loc", ""),
        "type_c": crop_data.get("type", ""),
        "export": crop_data.get("export", "")
    }
    return jsonify(context)

def generate_price_trend_chart(name, previous_x, previous_y, forecast_x, forecast_y):
    """Generate and save price trend visualization"""
    if not os.path.exists("visualization"):
        os.makedirs("visualization")
    
    image_filename = f"visualization/{name}_trend.png"
    
    plt.figure(figsize=(12,6))
    plt.plot(previous_x, previous_y, 'b-o', label="Historical Prices")
    plt.plot(forecast_x, forecast_y, 'r--o', label="Forecasted Prices")
    plt.xticks(rotation=45)
    plt.xlabel("Month/Year")
    plt.ylabel("Price (₹/quintal)")
    plt.title(f"{name.capitalize()} Price Trends")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(image_filename)
    plt.close()
    
    return f"/visualization/{name}_trend.png"

@app.route('/visualization/<path:filename>')
def serve_visualization(filename):
    return send_from_directory('visualization', filename)

def get_crop_data(name):
    """Mock function - replace with your actual data source"""
    return {
        "image": f"/static/images/{name}.jpg",
        "prime_loc": "Punjab, Haryana, UP",
        "type": "Rabi" if name.lower() in ['wheat', 'barley'] else "Kharif",
        "export": "Yes" if name.lower() in ['rice', 'wheat'] else "No"
    }

######################################################


def TopFiveWinners():
    """
    Returns top 5 commodities with highest month-over-month price increase percentage
    Format: [[name, current_price, percentage_change], ...]
    """
    try:
        current_date = datetime.now()
        current_month = current_date.month
        current_year = current_date.year
        prev_month = current_month - 1 if current_month > 1 else 12
        prev_year = current_year if current_month > 1 else current_year - 1
        
        # Get rainfall data (with bounds checking)
        current_rainfall = annual_rainfall[current_month - 1] if current_month <= len(annual_rainfall) else annual_rainfall[-1]
        prev_rainfall = annual_rainfall[prev_month - 1] if prev_month <= len(annual_rainfall) else annual_rainfall[-1]
        
        results = []
        
        for commodity in commodity_list:
            try:
                # Get current and previous month predictions
                current_price = commodity.predict_price(current_month, current_year, current_rainfall)
                prev_price = commodity.predict_price(prev_month, prev_year, prev_rainfall)
                
                # Calculate percentage change (handle division by zero)
                if prev_price == 0:
                    percent_change = 0.0
                else:
                    percent_change = ((current_price - prev_price) / prev_price) * 100
                
                crop_name = commodity.getCropName()
                results.append({
                    'name': crop_name,
                    'current_price': round(current_price, 2),
                    'percent_change': round(percent_change, 2)
                })
                
            except Exception as e:
                print(f"Error processing {commodity.name}: {str(e)}")
                continue
        
        # Sort by percentage change descending and take top 5
        top_winners = sorted(results, key=lambda x: x['percent_change'], reverse=True)[:5]
        
        return top_winners
    
    except Exception as e:
        print(f"Error in TopFiveWinners: {str(e)}")
        return []

def TopFiveLosers():
    """
    Returns top 5 commodities with highest month-over-month price decrease percentage
    Format: [[name, current_price, percentage_change], ...]
    """
    try:
        current_date = datetime.now()
        current_month = current_date.month
        current_year = current_date.year
        prev_month = current_month - 1 if current_month > 1 else 12
        prev_year = current_year if current_month > 1 else current_year - 1
        
        # Get rainfall data (with bounds checking)
        current_rainfall = annual_rainfall[current_month - 1] if current_month <= len(annual_rainfall) else annual_rainfall[-1]
        prev_rainfall = annual_rainfall[prev_month - 1] if prev_month <= len(annual_rainfall) else annual_rainfall[-1]
        
        results = []
        
        for commodity in commodity_list:
            try:
                # Get current and previous month predictions
                current_price = commodity.predict_price(current_month, current_year, current_rainfall)
                prev_price = commodity.predict_price(prev_month, prev_year, prev_rainfall)
                
                # Calculate percentage change (handle division by zero)
                if prev_price == 0:
                    percent_change = 0.0
                else:
                    percent_change = ((current_price - prev_price) / prev_price) * 100
                
                crop_name = commodity.getCropName()
                results.append({
                    'name': crop_name,
                    'current_price': round(current_price, 2),
                    'percent_change': round(percent_change, 2)
                })
                
            except Exception as e:
                print(f"Error processing {commodity.name}: {str(e)}")
                continue
        
        # Sort by percentage change ascending and take top 5
        top_losers = sorted(results, key=lambda x: x['percent_change'])[:5]
        
        return top_losers
    
    except Exception as e:
        print(f"Error in TopFiveLosers: {str(e)}")
        return []

def SixMonthsForecast():
    """
    Returns 6-month price forecast for all commodities
    Format: {
        "commodity1": [(month_year, price), ...],
        "commodity2": [(month_year, price), ...],
        ...
    }
    """
    try:
        current_date = datetime.now()
        forecast = {}
        
        for commodity in commodity_list:
            try:
                commodity_forecast = []
                for i in range(1, 7):  # Next 6 months
                    month = (current_date.month + i - 1) % 12 or 12
                    year = current_date.year + (current_date.month + i - 1) // 12
                    rainfall = annual_rainfall[month-1] if month <= len(annual_rainfall) else annual_rainfall[-1]
                    
                    price = commodity.predict_price(month, year, rainfall)
                    commodity_forecast.append((f"{month}/{year}", round(price, 2)))
                
                forecast[commodity.getCropName()] = commodity_forecast
            except Exception as e:
                print(f"Error forecasting for {commodity.name}: {str(e)}")
                continue
        
        return forecast
    
    except Exception as e:
        print(f"Error in SixMonthsForecast: {str(e)}")
        return {}

#Api Route 
@app.route('/')
def index():
    context = {
        "top5": TopFiveWinners(),
        "bottom5": TopFiveLosers(),
    }
    return jsonify(context)





if __name__ == '__main__':
    app.run(debug=True)