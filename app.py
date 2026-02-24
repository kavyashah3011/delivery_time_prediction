from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

app = FastAPI()

# Templates & Static
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# LOAD & TRAIN MODEL
# =========================

df = pd.read_csv("Food_Delivery_Times.csv")

df = df.drop(columns=['Order_ID', 'Courier_Experience_yrs', 'Time_of_Day'])
df['Weather'] = df['Weather'].fillna(df['Weather'].mode()[0])
df['Traffic_Level'] = df['Traffic_Level'].fillna(df['Traffic_Level'].mode()[0])

df = pd.get_dummies(
    df,
    columns=['Weather', 'Traffic_Level', 'Vehicle_Type'],
    drop_first=True
)

X = df.drop('Delivery_Time_min', axis=1)
y = df['Delivery_Time_min']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
numeric_cols = ['Distance_km', 'Preparation_Time_min']

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

model = LinearRegression()
model.fit(X_train, y_train)

# Metrics
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# =========================
# REQUEST MODEL
# =========================

class InputData(BaseModel):
    distance: float
    prep_time: float
    weather: str
    traffic: str
    vehicle: str


# =========================
# ROUTES
# =========================

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/metrics")
async def metrics():
    return {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "r2": round(r2, 4)
    }


@app.post("/predict")
async def predict(data: InputData):

    user_df = pd.DataFrame(columns=X.columns)
    user_df.loc[0] = 0

    user_df.loc[0, 'Distance_km'] = data.distance
    user_df.loc[0, 'Preparation_Time_min'] = data.prep_time

    weather_col = "Weather_" + data.weather
    traffic_col = "Traffic_Level_" + data.traffic
    vehicle_col = "Vehicle_Type_" + data.vehicle

    if weather_col in user_df.columns:
        user_df.loc[0, weather_col] = 1
    if traffic_col in user_df.columns:
        user_df.loc[0, traffic_col] = 1
    if vehicle_col in user_df.columns:
        user_df.loc[0, vehicle_col] = 1

    user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])
    prediction = model.predict(user_df)

    return {"prediction": float(prediction[0])}


# =========================
# RUN FOR RAILWAY
# =========================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
