from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


model = joblib.load("aqi_model.pkl")

# Get the exact feature names used during training (important!)
FEATURES = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else []

# Labels + units for common AQI features (anything unknown will still work)
META = {
    "CO(GT)": ("CO", "mg/m³"),
    "NOx(GT)": ("NOx", "ppb"),
    "NO2(GT)": ("NO2", "ppb"),
    "O3(GT)": ("O3", "ppb"),
    "SO2(GT)": ("SO2", "ppb"),
    "PM2.5": ("PM2.5", "µg/m³"),
    "PM10": ("PM10", "µg/m³"),
    "Temperature": ("Temperature", "°C"),
    "Humidity": ("Humidity", "%"),
    "Pressure": ("Pressure", "hPa"),
    "WindSpeed": ("Wind Speed", "m/s"),
    "WindDirection": ("Wind Direction", "degrees (0–360)"),
    "Hour": ("Hour", "0–23"),
    "DayOfWeek": ("Day of Week", "0–6"),
}

# Sample values
SAMPLE = {
    "CO(GT)": 4.0,
    "NOx(GT)": 180,
    "NO2(GT)": 90,
    "O3(GT)": 60,
    "SO2(GT)": 20,
    "PM2.5": 55,
    "PM10": 110,
    "Temperature": 30,
    "Humidity": 60,
    "Pressure": 1012,
    "WindSpeed": 3,
    "WindDirection": 180,
    "Hour": 12,
    "DayOfWeek": 3
}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predictpage")
def predict_page():
    return render_template("predict.html", features=FEATURES, meta=META, sample=SAMPLE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(request.form[f]) for f in FEATURES]
        pred = float(model.predict([values])[0])
        return render_template("predict.html", features=FEATURES, meta=META, sample=SAMPLE, prediction=round(pred, 2))
    except Exception as e:
        return render_template("predict.html", features=FEATURES, meta=META, sample=SAMPLE, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)