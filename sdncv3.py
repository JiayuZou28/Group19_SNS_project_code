import pandas as pd

file_path = "C:/Users/aoc/ansel/real/birmingham 2025-03-01 to 2025-03-18.csv" # Define birmingham file path

df = pd.read_csv(file_path, usecols=["datetime","tempmax", "tempmin", "temp"])# Read CSV,only read B,C、D、E columns

print(df)


import os
# Save as csv file
default_dir = os.getcwd()
save_path = os.path.join(default_dir, "birmingham_data.csv")# Get the default directory


df.to_csv(save_path, index=False, encoding="utf-8")# Save data in csv file
print(f" file saves at: {save_path}")


import pandas as pd

file_path_1 = "C:/Users/aoc/ansel/real/london 2025-03-01 to 2025-03-18.csv"# Define london file path

df = pd.read_csv(file_path_1, usecols=["datetime","tempmax", "tempmin", "temp"])# Read CSV,only read B,C、D、E columns

print(df)


import os
# Save as csv file

default_dir = os.getcwd()
save_path_1 = os.path.join(default_dir, "london_data.csv")# Get the default directory


df.to_csv(save_path_1, index=False, encoding="utf-8")# Save data in csv file
print(f" file saves at: {save_path_1}")


import os

directory = "C:/Users/aoc/ansel/real/" # Specify directory
all_files = os.listdir(directory) 
csv_files = [file for file in all_files if file.endswith(".csv")] # Only csv file 

print("CSV files:", csv_files)


## Step 1: Load two csv files


import pandas as pd

london_file = "C:/Users/aoc/ansel/real/london_data.csv" # My own liptop file path
birmingham_file = "C:/Users/aoc/ansel/real/birmingham_data.csv" # My own liptop file path

london_df = pd.read_csv(london_file, parse_dates=["datetime"], index_col="datetime")[['tempmax', 'tempmin', 'temp']]
birmingham_df = pd.read_csv(birmingham_file, parse_dates=["datetime"], index_col="datetime")[['tempmax', 'tempmin', 'temp']]
# Read "tempmax,tempmin,temp"three columns of csv files

print("London Dataset :\n", london_df.head(18))
print("Birmingham Dataset :\n", birmingham_df.head(18))


## Step 2: Data preprocessing


import numpy as np
from sklearn.preprocessing import MinMaxScaler


seq_length = 14 # Last 14 days forecast day 15

scaler = MinMaxScaler() # Normalised data


def prepare_data(df):
    
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index) # Normalisation
    
    # Processing time series data
    X, y = [], []
    for i in range(len(df_scaled) - seq_length):
        X.append(df_scaled.iloc[i:i+seq_length].values)
        y.append(df_scaled.iloc[i+seq_length].values)  
    
    return np.array(X), np.array(y), scaler  # The normaliser use to inverse normalise the results


X_london, y_london, london_scaler = prepare_data(london_df)
X_birmingham, y_birmingham, birmingham_scaler = prepare_data(birmingham_df)
# Processing London and Birmingham data


# Delineate London train and test datasets
train_size_london = int(len(X_london) * 0.8)
X_train_london, X_test_london = X_london[:train_size_london], X_london[train_size_london:]
y_train_london, y_test_london = y_london[:train_size_london], y_london[train_size_london:]

# Delineate Birmingham train and test datasets
train_size_birmingham = int(len(X_birmingham) * 0.8)
X_train_birmingham, X_test_birmingham = X_birmingham[:train_size_birmingham], X_birmingham[train_size_birmingham:]
y_train_birmingham, y_test_birmingham = y_birmingham[:train_size_birmingham], y_birmingham[train_size_birmingham:]



print("X_train_london shape:", X_train_london.shape)
print("y_train_london shape:", y_train_london.shape)
print("X_train_birmingham shape:", X_train_birmingham.shape)
print("y_train_birmingham shape:", y_train_birmingham.shape)

print(" London shape", london_df.shape)
print(" Birmingham shape ", birmingham_df.shape)
# Print data shape (fix 1)


def prepare_data(df):

    print(f" Raw dataset size: {df.shape}")  # Display the size of the dataset
    if df.shape[0] < seq_length:
        print(" Less data samples to create training samples!")# Check if data samples are sufficient to create training samples
        return np.array([]), np.array([]), None

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(df_scaled) - seq_length):
        X.append(df_scaled[i:i+seq_length])  
        y.append(df_scaled[i+seq_length])  # Predict day 15 tempmax,tempmin, temp

    X = np.array(X)
    y = np.array(y)

    print(f"Generate data successfull. X shape: {X.shape}, y shape: {y.shape}")  # Should be (Sample size, 17, 3)

    return X, y, scaler


## Step 3: Build LSTM model


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam


# London LSTM model
def build_lstm_model():
    model = Sequential([
        LSTM(100, activation="tanh", return_sequences=True, input_shape=(seq_length, 3)),
        LSTM(100, activation="relu"),
        Dense(3)  # Predict tempmax, tempmin, temp
    ])
    model.compile(optimizer="adam", loss=MeanSquaredError())
    return model


# Birmingham LSTM model
def build_model_birmingham(seq_length=14, lstm_units=128, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        LSTM(lstm_units, activation="tanh", return_sequences=True, input_shape=(seq_length, 3)),
        Dropout(dropout_rate),
        
        LSTM(lstm_units, activation="relu"),
        Dropout(dropout_rate),
        
        Dense(3)  # Predict tempmax, tempmin, temp
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError())
    return model


## Step4: Train the models


# Training the London model
model_london = build_lstm_model()
model_london.fit(X_train_london, y_train_london, epochs=40, batch_size=16, validation_data=(X_test_london, y_test_london))

# Training the Birmingham model
model_birmingham = build_lstm_model()
model_birmingham.fit(X_train_birmingham, y_train_birmingham, epochs=40, batch_size=16, validation_data=(X_test_birmingham, y_test_birmingham))

model_birmingham_v2 = build_model_birmingham(seq_length=14, lstm_units=128, dropout_rate=0.2, learning_rate=0.001)
model_birmingham_v2.save("oracle_birmingham_v2.h5")

# Save the models
model_london.save("oracle_london.h5")
model_birmingham.save("oracle_birmingham.h5")

print("Prediction models are trained and saved!")


## Step 5: Deploying the Flask Server


from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

app = Flask(__name__)


# Load the trained LSTM models
model_london = load_model("oracle_london.h5",custom_objects={"MeanSquaredError": MeanSquaredError()})
model_birmingham = load_model("oracle_birmingham_v2.h5",custom_objects={"MeanSquaredError": MeanSquaredError()})


def predict_temperature(model, last_days, scaler):

    last_days = np.array(last_days).reshape(1, seq_length, 3)  # Need to reshape to (1, 14, 3)

    prediction_scaled = model.predict(last_days)
    prediction = scaler.inverse_transform(prediction_scaled).flatten()  # Inverse normalisation

 
    prediction = prediction.astype(float).tolist()# Convert to Python float type

    print("Prediction result:", prediction)
    return {"tempmax": prediction[0], "tempmin": prediction[1], "temp": prediction[2]}



@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]
    city = request.json["city"].lower()

    if city == "london":
        prediction = predict_temperature(model_london, data, london_scaler)
    elif city == "birmingham":
        prediction = predict_temperature(model_birmingham, data, birmingham_scaler)
    else:
        return jsonify({"error": "City not supported"}), 400

    return jsonify({"city": city.capitalize(), "prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5062)







