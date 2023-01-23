# Run this app with:
#  >conda activate base
#  >pip install dash (if not done previously)
#  >python dash_basic.py
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import pymssql

# for later iterations:
# import pymssql

from config import database
from config import table
from config import username
from config import password
from config import server
import random
from dash.dependencies import Input, Output

import joblib

path_2 = "C:/Users/wsutt/OneDrive/Desktop/devten/repos/intro-kafka/demo-pickle-2/model-0.model"
loaded_model = joblib.load(path_2)

def update_prediction(X=None):
    yhat = loaded_model.predict([[X]])
    return yhat

print('script loading, model gets...')
y = update_prediction(2000)
print(y)

app = dash.Dash(__name__)



def get_data_sql():
    conn = pymssql.connect(server,username, password, database)
    cursor = conn.cursor()
    query = f"SELECT * FROM {table}"
    df = pd.read_sql(query, conn)
    cols = ['Weight', 'MPG.city', 'MPG.highway']
    for col in cols:
        df[col] = df[col].astype('int')


def get_data():
    df = pd.read_csv('data/Cars93.csv')
    return df

def get_data_sample():
    i = random.randint(0,2)
    df = pd.read_csv(f'data/sample-{i}.csv')
    return df

def make_figure_1():
    df = get_data_sample()
    df2 = df[['Weight', 'MPG.city', 'MPG.highway']]
    fig = px.scatter(df2, x='Weight', y='MPG.city', title='Simple Figure 1')
    return fig

def make_figure_2(opacity=0.5):
    df = get_data()
    df2 = df[['Weight', 'MPG.city', 'MPG.highway']]
    fig = px.scatter(
            df2, 
            x='Weight', 
            y='MPG.city', 
            title='User update',
            opacity=opacity,
            )
    return fig

def make_figure_predict(value=None):
    df = get_data()
    fig = px.bar(pd.DataFrame({'mpg':[31,30]}))
    return fig


app.layout = html.Div(children=[
    html.H1(children='Hello Dash. Wills Dashboard!.'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='first-graph',
        figure=make_figure_1(),
    ),

    dcc.Interval(
        id='interval-component',
        interval=5*1000,
        n_intervals=0,
    ),

    dcc.Input(
        value="2000",
        id='car-weight-input'
    ),

    dcc.Graph(
            id='example-graph',
            figure=make_figure_predict()
    ),
])


@app.callback(
    Output('first-graph', 'figure'),
    Input('interval-component', 'n_intervals'),
)
def update_figure_1(passin_value):
    return make_figure_1()

@app.callback(
    Output('example-graph', 'figure'),
    Input('car-weight-input', 'value'),
)
def update_figure_predict(weight_value):
    print(weight_value)
    try:
        weight_float = float(weight_value)
    except:
        weight_float = 2000.0
    
    try: 
        yhat = update_prediction(weight_float)
        print(yhat)
    except Exception as e:
        print(e)
    # yhat = 40
    
    fig = px.bar(pd.DataFrame({'mpg':[yhat[0],30]}))
    return fig


# run the app
if __name__ == '__main__':
    app.run_server(debug=True )
