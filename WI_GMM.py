# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:59:20 2024

@author: 35191
"""

import os
# Set working directory
os.chdir(r"D:\My projects\my thesis\PhD\Thesis\Chapter 4\Alvalade\codes\ANN-Ground-Motion-Model-for-Western-Iberia-main")
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')  # Set TensorFlow logger to only show errors
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')  # Ignore Keras UserWarnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.base')
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)




def calculate_psa_TEA24(case, Mw, Rjb, Depth, FM):
    """
    Calculate PSA, Period, and additional parameters for given earthquake parameters.

    Parameters:
    - case: int, placeholder case ID.
    - Mw: float, earthquake magnitude.
    - Rjb: float, distance in km.
    - Depth: float, hypocenter depth in km.
    - FM: int, fault mechanism.

    Returns:
    - DataFrame with 'Period', 'PSA', 'between_event', 'within_event', and 'total_sigma' values.
    """
    
    # Load model and scalers
    GMP_model_properties = pd.read_csv('Regression_metrics.csv')
    NN_model = tf.keras.models.load_model('ANN_Portugal_rock.h5')
    Mw_scaler = joblib.load("Mw_scaler.pkl")
    Rjb_scaler = joblib.load("Rjb_scaler.pkl")
    Depth_scaler = joblib.load("Depth_scaler.pkl")
    
    # Rescale input data
    Mw_scaled = Mw_scaler.transform(np.array([[Mw]]))
    Rjb_scaled = Rjb_scaler.transform(np.array([[Rjb]]))
    Depth_scaled = Depth_scaler.transform(np.array([[Depth]]))

    # Prepare input array for model
    Input_data = np.column_stack((case, Mw_scaled[0], Rjb_scaled[0], Depth_scaled[0], FM)).reshape(1, -1)

    # Predict PSA values
    Median_GM = np.exp(NN_model.predict(Input_data))  # Assuming output is in log scale

    # Convert predictions to DataFrame for easier handling
    Median_GM = pd.DataFrame(Median_GM, columns=GMP_model_properties.IM, dtype=float)

    # Optional: Drop unnecessary columns if specified
    Median_GM.drop(['PGV'], axis=1, inplace=True, errors='ignore')

    # Define additional data inside the function
    additional_data = pd.DataFrame({
        'between_event': [0.519290669, 0.524470346, 0.529756778, 0.529688794, 0.532506304, 0.530049414, 0.532964882, 
                          0.53110028, 0.536437409, 0.539748677, 0.539697967, 0.540025487, 0.547286421, 0.552996317, 
                          0.571526251, 0.581546766, 0.594193527, 0.586797102, 0.560534959, 0.536437409],
        'residual': [0.264047911, 0.260283973, 0.251218468, 0.244845183, 0.241076109, 0.237037624, 0.23417217, 
                     0.232926235, 0.23273434, 0.234710867, 0.238736016, 0.240160935, 0.24949612, 0.251870684, 
                     0.25531389, 0.256991427, 0.257851675, 0.250420973, 0.243669106, 0.249203819],
        'total_sigma': [0.582566818, 0.585505671, 0.586304496, 0.583540386, 0.584534562, 0.580636905, 0.582141023, 
                        0.579932874, 0.584748122, 0.588572702, 0.590143017, 0.591020136, 0.601473806, 0.607654316, 
                        0.625961211, 0.635799681, 0.647729445, 0.637998042, 0.611207063, 0.584748122],
        'IM': ['f=0.25', 'f=0.32', 'f=0.5', 'f=0.67', 'f=0.8', 'f=1', 'f=1.3', 'f=1.6', 'f=2', 'f=2.5', 'f=4', 
               'f=5', 'f=8', 'f=10', 'f=15', 'f=20', 'f=25', 'f=50', 'PGA', 'PGV']
    })

    # Initialize lists for frequencies and PSA values
    frequencies = []
    psa_values = []
    between_event_values = []
    within_event_values = []
    total_sigma_values = []

    # Process columns and handle 'PGA' as a special case
    for col in Median_GM.columns:
        if col == 'PGA':
            frequencies.append(np.inf)  # Set PGA to an effectively infinite frequency, equivalent to period=0
            psa_values.append(Median_GM[col].iloc[0])  # Add PGA value directly
            additional_row = additional_data[additional_data['IM'] == 'PGA']
        elif '=' in col:
            try:
                freq_value = float(col.split('=')[1])  # Extract the frequency value after '='
                frequencies.append(freq_value)
                psa_values.append(Median_GM[col].iloc[0])
                additional_row = additional_data[additional_data['IM'] == col]
            except ValueError:
                print(f"Skipping non-numeric column: {col}")
                continue

        # Add corresponding additional parameters if found
        if not additional_row.empty:
            between_event_values.append(additional_row['between_event'].values[0])
            within_event_values.append(additional_row['residual'].values[0])
            total_sigma_values.append(additional_row['total_sigma'].values[0])

    # Calculate periods from frequencies (setting period of 'PGA' as 0)
    periods = [1.0 / f if f != np.inf else 0 for f in frequencies]  # If f is inf (PGA), period is 0

    # Convert PSA values to numpy array for unit conversion if needed
    psa_values = np.array(psa_values) / 981  # Convert from cm/s² to g

    # Create a DataFrame with Period, PSA, and additional parameters
    psa_df = pd.DataFrame({
        'Period': periods,
        'PSA': psa_values,
        'between_event': between_event_values,
        'within_event': within_event_values,
        'total_sigma': total_sigma_values
    })
    
    return psa_df

# Example usage
case = 0
Mw = 6.0
Rjb = 10.0
Depth = 20.0
FM = 0

# Call the function and get PSA and Period data with additional parameters
psa_df = calculate_psa_TEA24(case, Mw, Rjb, Depth, FM)

# Display the result
print(psa_df)













import math

def haversine(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Calculate the differences between the coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance in kilometers
    distance = R * c
    return distance

# Coordinates
rupture_lat = 38.8
rupture_lon = -9.40
site_lat = 38.750573
site_lon = -9.1672814

# Calculate distance
distance_km = haversine(rupture_lat, rupture_lon, site_lat, site_lon)
print(f"Distance: {distance_km:.2f} km")
