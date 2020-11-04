from sqlalchemy import create_engine
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

# Label encoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, normalize, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

def read_file():
    engine = create_engine('sqlite:///data/traffic.db')
    df = pd.read_sql_query("SELECT * FROM traffic WHERE date_time >= '2013-01-01 00:00:00'", engine)
    return df

def preprocess(df):
    df['date_time'] = pd.to_datetime(df.date_time)
    df['year']= df.date_time.dt.year
    df['month']= df.date_time.dt.month
    df['week']= df.date_time.dt.weekofyear
    df['hour'] = df.date_time.dt.hour
    
    df['dayofmonth']= df.date_time.dt.day
    df['dayofweek']= df.date_time.dt.dayofweek
    
    df['workday'] = df.holiday == 'None'
    df.workday = df.workday.astype('int')
    return df

def fit_predict_model(df):
   
    le = LabelEncoder()

# Create new features
    df['holiday_enc'] = le.fit_transform(df['holiday'])
    df['weather_main_enc'] =  le.fit_transform(df['weather_main'])
    df['weather_description_enc'] =  le.fit_transform(df['weather_description'])

    features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'dayofweek', 
   'holiday_enc', 'weather_main_enc', 'weather_description_enc', 
           'year','month', 'week', 'dayofmonth', 'workday']

    num_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'dayofweek', 
          'holiday_enc', 'year','month', 'week', 'dayofmonth', 'workday']
    
    train, test = train_test_split(df, test_size=0.3,  random_state=123)
    TEXT_COLUMNS = ['weather_main', 'weather_description']
    df[TEXT_COLUMNS]
    
    def combine_text_columns(data_frame, to_add=TEXT_COLUMNS):  
        text_data = data_frame[TEXT_COLUMNS]
    # Join all text items in a row that have a space in between
        return text_data.apply(lambda x: " ".join(x), axis=1)

    combine_text_columns(df)
    
    get_text_data = FunctionTransformer(combine_text_columns, validate=False)

# Obtain the numeric data: get_numeric_data
    get_numeric_data = FunctionTransformer(lambda x: x[num_features], validate=False)

# Create a FeatureUnion with nested pipeline: process_and_join_features
    process_and_join_features = FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data)
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(ngram_range=(1, 2)))
                ]))
             ]
        )

# Instantiate nested pipeline: pl
    pl = Pipeline([
        ('union', process_and_join_features),
        ('GradientBoostingRegressor', GradientBoostingRegressor(max_depth = 9 ))
    ])


# Fit pl to the training data
    pl.fit(train, train.traffic_volume)
    test['pred'] = pl.predict(test)
    
    return test

def plotgraph(test):
    # Data
    df=pd.DataFrame({'x': range(20), 'y1': test.traffic_volume.head(20), 'y2': test.pred.head(20) })
# multiple line plot
    plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4, label="True Value")
    plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=2, label="pred" )
    plt.legend()

def print_metrics(test):    
    # Measure the local RMSE
    rmse = sqrt(mean_squared_error(test['traffic_volume'], test['pred']))
    print('RMSE for Baseline II - NLP model: {:.3f}'.format(rmse))
    print(test[['traffic_volume', 'pred']].head(10))

def main_run():
    df = read_file()
    df = preprocess(df)
    test = fit_predict_model(df)
    plotgraph(test)
    print_metrics(test)

main_run()



