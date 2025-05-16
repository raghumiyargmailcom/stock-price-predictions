from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

app = Flask(__name__)

# Load models and scalers
aapl_model = joblib.load("models/lstm_aapl_model_full.pkl")
aapl_scaler = joblib.load("models/lstm_aapl_scaler.pkl")
amzn_model = joblib.load("models/xgb_amzn_model.pkl")
amzn_scaler = joblib.load("models/xgb_amzn_scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    selected_stock = None
    plot_url = None
    error = None
    target_date_str = None

    if request.method == "POST":
        selected_stock = request.form.get("stock")
        target_date_str = request.form.get("target_date")

        if not target_date_str:
            return render_template("index.html", error="Please select a date.")

        try:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
        except ValueError:
            return render_template("index.html", error="Invalid date format.")

        if selected_stock == "AAPL":
            df = pd.read_csv("data/AAPL_Historical_Data_Upto_2024.csv")
            df["Date"] = pd.to_datetime(df["Date"])
            df.sort_values("Date", inplace=True)
            
            # Adjust for stock split
            split_date = pd.to_datetime("2022-07-18")
            df["Close"] = df.apply(
                lambda row: row["4. close"] / 20 if row["Date"] < split_date else row["4. close"], axis=1
            )
            
            last_known_prices = df["Close"].values[-3:].tolist()
            start_date = df["Date"].iloc[-1] + timedelta(days=1)
            days_to_predict = (target_date - start_date).days + 1

            if days_to_predict <= 0:
                return render_template("index.html", error="Please choose a date after the last available date.")

            predictions = []
            dates = []
            for i in range(days_to_predict):
                input_features = np.array(last_known_prices[-3:]).reshape(1, -1)
                input_scaled = aapl_scaler.transform(input_features)
                next_price = aapl_model.predict(input_scaled)[0]
                last_known_prices.append(next_price)
                predictions.append(next_price)
                dates.append(start_date + timedelta(days=i))

            target_prediction = round(predictions[-1], 2)
            prediction = target_prediction
            past_df = df[["Date", "Close"]].copy()
            past_df.rename(columns={"Close": "4. close"}, inplace=True)

        elif selected_stock == "AMZN":
            df = pd.read_csv("data/AMZN_Historical_Data_Upto_2024.csv")
            df["Date"] = pd.to_datetime(df["Date"])
            df.sort_values("Date", inplace=True)

            df["Close"] = df["4. close"]
            last_known_prices = df["Close"].values[-3:].tolist()
            start_date = df["Date"].iloc[-1] + timedelta(days=1)
            days_to_predict = (target_date - start_date).days + 1

            if days_to_predict <= 0:
                return render_template("index.html", error="Please choose a date after the last available date.")

            predictions = []
            dates = []
            for i in range(days_to_predict):
                input_features = np.array(last_known_prices[-3:]).reshape(1, -1)
                input_scaled = amzn_scaler.transform(input_features)
                next_price = amzn_model.predict(input_scaled)[0]
                last_known_prices.append(next_price)
                predictions.append(next_price)
                dates.append(start_date + timedelta(days=i))

            target_prediction = round(predictions[-1], 2)
            prediction = target_prediction
            past_df = df[["Date", "Close"]].copy()
            past_df.rename(columns={"Close": "4. close"}, inplace=True)

        else:
            return render_template("index.html", error="Invalid stock selection")

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(past_df["Date"], past_df["4. close"], label="Actual", marker="o")
        ax.axvline(x=target_date, color='red', linestyle='--', label='Prediction Date')
        ax.scatter(target_date, prediction, color='green', label='Predicted Price', s=100)
        ax.set_title(f"{selected_stock} - Predicted Price on {target_date_str}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode("utf-8")
        plot_url = f"data:image/png;base64,{plot_data}"
        plt.close()

    return render_template("index.html", prediction=prediction, stock=selected_stock,
                           plot_url=plot_url, error=error, date=target_date_str)

if __name__ == "__main__":
    app.run(debug=True)
