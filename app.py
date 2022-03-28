import argparse
import pandas as pd
import numpy as np
from fbprophet import Prophet
import datetime

# You can write code above the if-main block.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    df = pd.read_csv(args.training)
    X = df.rename(columns={'日期':'ds', '備轉容量(MW)':'y'})[['ds', 'y']]
    X['ds'] = pd.to_datetime(X['ds'], format='%Y%m%d')

    model = Prophet(yearly_seasonality=True, daily_seasonality=True)
    model.fit(X)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    forecast = forecast.loc[datetime.datetime(2022, 3, 29) < forecast['ds'], ['ds','yhat']]
    forecast = forecast.loc[forecast['ds'] < datetime.datetime(2022, 4, 14), ['ds','yhat']]
    forecast['yhat'] = np.round(forecast['yhat']).astype(np.int32)
    forecast['ds'] = forecast['ds'].dt.strftime('%Y%m%d')
    forecast = forecast.rename(columns={'ds':'date', 'yhat':'operating_reserve(MW)'})
    forecast.to_csv(args.output, index=False)