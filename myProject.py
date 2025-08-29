import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

WAQI_API_TOKEN = ""  # Insert your WAQI token here
LAT, LON = 25.6, 85.1
HISTORY_HOURS = 24
DT_MIN = 15
LAG = 8
HORIZON = 4

# Fetch live PM2.5 from WAQI
def fetch_pm25_waqi(lat, lon, token):
    try:
        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/"
        r = requests.get(url, params={"token": token}, timeout=5)
        r.raise_for_status()
        data = r.json()
        if data["status"] == "ok":
            pm25 = data["data"].get("iaqi", {}).get("pm25", {}).get("v", None)
            return float(pm25) if pm25 is not None else None
    except:
        return None

# Basic solar and pm25 signal simulation
def simulate_signals(periods, dt_min=15):
    np.random.seed(42)
    start = datetime.now(timezone.utc) - timedelta(minutes=dt_min * periods)
    times = [start + timedelta(minutes=dt_min*i) for i in range(periods)]
    hours = np.array([t.hour + t.minute/60 for t in times])

    solar = 5 * np.maximum(0, np.sin((hours - 6) / 12 * np.pi)) + np.random.normal(0, 0.2, periods)
    solar = np.clip(solar, 0, None)

    traffic = 0.5 + 0.4 * np.exp(-0.5 * ((hours - 8)/1.5)**2) + 0.4 * np.exp(-0.5 * ((hours - 18)/1.5)**2)
    time_h = np.arange(periods) * (dt_min / 60)
    pm25 = 30 + 20 * traffic + 6 * np.sin(2 * np.pi * time_h / 24) + np.random.normal(0, 4, periods)
    pm25 = np.clip(pm25, 1, None)

    return times, solar, pm25

# Build lagged features for forecasting
def make_lagged(series, lag=LAG, horizon=HORIZON):
    arr = np.array(series)
    X, y = [], []
    for i in range(len(arr) - lag - horizon + 1):
        X.append(arr[i:i+lag])
        y.append(arr[i+lag+horizon-1])
    return np.array(X), np.array(y)

# Categorize PM2.5 levels into air quality categories
def categorize_pm25(val):
    if val <= 15: return "Good 😊"
    elif val <= 35: return "Moderate 🙂"
    elif val <= 55: return "Unhealthy for Sensitive 😷"
    elif val <= 150: return "Unhealthy ⚠"
    else: return "Hazardous 🚨"

# Generate alert message based on PM2.5 value
def generate_alert(pm25_value):
    category = categorize_pm25(pm25_value)
    alert_msg = f"Air Quality Alert!!!\n" \
                f"Predicted PM2.5: {pm25_value:.1f} µg/m³\n" \
                f"Category: {category}\n"

    actions = {
        "Good 😊": "Air quality is good. Enjoy your day!",
        "Moderate 🙂": "Air quality is moderate. Sensitive individuals should take care.",
        "Unhealthy for Sensitive 😷": "Sensitive groups should reduce outdoor activities.",
        "Unhealthy ⚠": "Limit outdoor exertion. Consider wearing masks.",
        "Hazardous 🚨": "Avoid outdoor activities. Follow all pollution control measures."
    }

    alert_msg += "Recommended action: " + actions.get(category, "")
    return alert_msg

# === MAIN ===
periods = int(HISTORY_HOURS * 60 / DT_MIN)
times, solar, pm25_sim = simulate_signals(periods, DT_MIN)
live_pm25 = fetch_pm25_waqi(LAT, LON, WAQI_API_TOKEN)

if live_pm25 is not None:
    pm25_sim[-1] = live_pm25
    print("Using live PM2.5 from WAQI.")

df = pd.DataFrame({"time": times, "solar_kw": solar, "pm25": pm25_sim}).set_index("time")

X, y = make_lagged(df["pm25"], lag=LAG, horizon=HORIZON)
split = int(0.75 * len(X))

X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"PM2.5 forecast RMSE: {rmse:.2f} µg/m³")

latest_window = df["pm25"].values[-LAG:].reshape(1, -1)
next_pm25 = model.predict(latest_window)[0]

print(f"Next 1-hour PM2.5 prediction: {next_pm25:.1f} µg/m³")
print("Air quality category:", categorize_pm25(next_pm25))

# Generate and print alert based on forecast
alert_message = generate_alert(next_pm25)
print("\n" + alert_message)

# Plot results
plt.figure(figsize=(10,4))
plt.plot(df.index[-len(y_test):], y_test, label="True PM2.5")
plt.plot(df.index[-len(y_test):], y_pred, '--', label="Predicted PM2.5")
plt.title("PM2.5 1-hour Ahead Forecast")
plt.xlabel("Time")
plt.ylabel("PM2.5 (µg/m³)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
