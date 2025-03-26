
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os


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

annual_rainfall = [34,	26,	42,	35,	57,	165	,320,	275	,195,	76,	44,14,]
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
        dataset = pd.read_csv(csv_name)
        
        # Assuming CSV format: [Month, Year, Rainfall, WPI]
        self.X = dataset.iloc[:, :-1].values  # Features: Month, Year, Rainfall
        self.Y = dataset.iloc[:, -1].values   # Target: WPI
        
        # Feature engineering
        self._add_seasonal_features()
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Train commodity-specific model
        self.regressor = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        self.regressor.fit(self.X_scaled, self.Y)
        
        # Verify model
        test_pred = self.regressor.predict(self.X_scaled[:1])
        print(f"{self.name} model initialized. Sample prediction: {test_pred[0]:.1f}")

    def _add_seasonal_features(self):
        """Add quarter and growth phase features"""
        quarters = []
        growth_phases = []
        for month in self.X[:, 0]:  # Month is first column
            quarters.append((month - 1) // 3 + 1)
            # Growth phases: 1=sowing, 2=growing, 3=harvest
            growth_phases.append(1 if month in [6,7] else (2 if month in [8,9,10] else 3))
        
        self.X = np.column_stack((
            self.X,
            quarters,
            growth_phases
        ))

    def predict_wpi(self, month, year, rainfall):
        """Predict WPI for specific conditions"""
        quarter = (month - 1) // 3 + 1
        growth_phase = 1 if month in [6,7] else (2 if month in [8,9,10] else 3)
        
        input_data = np.array([[month, year, rainfall, quarter, growth_phase]])
        scaled_input = self.scaler.transform(input_data)
        return self.regressor.predict(scaled_input)[0]

    def predict_price(self, month, year, rainfall):
        """Convert WPI prediction to actual price"""
        wpi = self.predict_wpi(month, year, rainfall)
        return self.base_price * (wpi / 100)  # Assuming base WPI=100




# Initialize all commodities
commodity_list = []
for name, path in commodity_dict.items():
    try:
        base_price = base[name.capitalize()]
        commodity_list.append(Commodity(path, base_price))
        print(f"Loaded {name} successfully")
    except Exception as e:
        print(f"Error loading {name}: {str(e)}")

# Example usage
def get_forecast(commodity_name, month, year, rainfall):
    for commodity in commodity_list:
        if commodity.name.lower() == commodity_name.lower():
            wpi = commodity.predict_wpi(month, year, rainfall)
            price = commodity.predict_price(month, year, rainfall)
            return {
                'commodity': commodity.name,
                'month': month,
                'year': year,
                'predicted_wpi': round(wpi, 2),
                'predicted_price': round(price, 2),
                'base_price': commodity.base_price
            }
    return None

# Test prediction
print(get_forecast("wheat", 8, 2024, 150))
print(get_forecast("barley", 9, 2024, 200))
print(get_forecast("arhar", 9, 2024, 200))

































# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split, cross_val_score
# import numpy as np
# import pandas as pd

# def train(csv_name):
#     dataset = pd.read_csv(csv_name)
#     X = dataset.iloc[:, :-1].values
#     Y = dataset.iloc[:, -1].values  # Ensure last column is used as target

#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=69)

#     # Fitting Random Forest Regression with optimized parameters
#     regressor = RandomForestRegressor(
#         n_estimators=500,  # Increase number of trees for better stability
#         max_depth=None,  # Allow trees to grow deeper
#         min_samples_split=4,  # Avoid overfitting
#         min_samples_leaf=2,  # Improve generalization
#         random_state=42,
#         n_jobs=-1  # Use all CPU cores for faster training
#     )
#     regressor.fit(X_train, Y_train)
    
#     y_pred_forest = regressor.predict(X_test)
    
#     # Print values in a formatted way
#     print(f"{'Index':<10}{'Actual Value':<15}{'Predicted Value':<15}")
#     print("-" * 40)
#     for i, (actual, predicted) in enumerate(zip(Y_test, y_pred_forest), start=1):
#         print(f"{i:<10}{actual:<15.2f}{predicted:<15.2f}")
    
#     # Perform cross-validation to assess model performance
#     scores = cross_val_score(regressor, X_train, Y_train, cv=5, scoring='r2', n_jobs=-1)
#     print("Cross-validation R² scores:", scores)
#     print("Mean R² score:", np.mean(scores))
    
#     return regressor  # Return trained model for further use

# # Example usage
# train("static/Bajra.csv")
