import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable, FormatTemplate
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import pandas as pd
from os import listdir, remove
import pickle
from time import sleep
from strategy import *
from backtest import *
from strategyData import *
import matplotlib.pyplot as plt

app = dash.Dash(__name__)

# Define the layout.
app.layout = html.Div([

    # Section title
    html.H1("Section 1: Model Backtest"),
    html.Div(
        [
            html.Label('Moving Average Size in Decision Tree, Logliner, KNN Model',
                       style={'padding': 500}),
            dcc.Slider(
                id='size',
                min=5, max=100,
                marks={days: str(days) for days in range(0, 101, 5)},
                value=50),
            #             "Window Size: ",
            #             dcc.Input(id='input-size', value='30', type='text'),
        ],
        # Style it so that the submit button appears beside the input.
        style={'display': 'inline-block'}
    ),
    dcc.RadioItems(
        options=[
            {'label': 'Logistic', 'value': 'Logistic'},
            {'label': 'Decision Tree', 'value': 'Decision Tree'},
            {'label': 'Loglinear', 'value': 'Loglinear'},
            {'label': 'KNN', 'value': 'KNN'}

        ],
        id='md',
        value='Logistic',
        labelStyle={'display': 'inline-block'}
    ),
    html.Div(

        html.Table(
            [html.Tr([
                html.Th('Start Date'),
                html.Th('End Date'),
                html.Th('Bloomberg Identifier'),
                html.Th('Starting Cash'),
                html.Th('')
            ])]+
            [html.Tr([
                html.Td(
                    dcc.DatePickerSingle(
                        id='start-date',
                        min_date_allowed=date(2000, 1, 1),
                        max_date_allowed=date.today(),
                        initial_visible_month=date.today(),
                        date=date.today(),
                        style={'text-align': 'center'}
                    )
                ),
                html.Td(
                    dcc.DatePickerSingle(
                        id='end-date',
                        min_date_allowed=date(2000,1,1),
                        max_date_allowed=date.today(),
                        initial_visible_month=date.today(),
                        date=date.today(),
                        style={'text-align': 'center'}
                    )
                ),
                html.Td(
                    dcc.Input(
                        id='bbg-identifier-1', type="text",
                        value="IVV US Equity",
                        style={'text-align': 'center', 'line-height': '40px', 'font-size': '20px'}
                    )
                ),
                html.Td(
                    dcc.Input(
                        id='starting-cash', type="number",
                        value=1000000,
                        style={'text-align': 'center', 'line-height': '40px', 'font-size': '20px'}
                    )
                ),
                html.Button('Run Backtest', id='submit-button', n_clicks=0,
                            style={'text-align': 'center', 'line-height': '40px', 'font-size': '20px'})
            ])]
        )
    ),
    # Line break
    html.Br(),
    # Div to hold the initial instructions and the updated info once submit is pressed
    html.Div(id='info', children='Enter a your preferred window size and model then press "submit".'),
    html.Div([
        # Candlestick graph goes here:
        dcc.Graph(id='yield-curve'),
        html.H2(
            'Trade Blotter',
            style={
                'display': 'inline-block', 'width': '55%',
                'text-align': 'center'
            }
        ),
        DataTable(
            id='blotter',
            fixed_rows={'headers': True},
            style_cell={'textAlign': 'center'},
            style_table={'height': '300px', 'overflowY': 'auto'}
        )],
        style = {'display': 'inline-block', 'width': '55%'}
    ),
    # Another line break
    html.Br(),
    # Section title
    html.H1("Section 2: Suggest a Trade for next day"),
    # Div to confirm what trade was made
    html.Div(id='info2', children='Here will show trade suggestion for next day.'),
    html.Div(id='caution', children='Action type, Trading Symbol, Amount, Order Type, Price(only for limit order)'),
    # Radio items to select buy or sell
    dcc.Input(id='actn', value='BUY', type='text'),
    # Text input for the currency pair to be traded
    dcc.Input(id='symb', value='IVV', type='text'),
    # Numeric input for the trade amount
    dcc.Input(id='amount', value=100, type='number'),
    dcc.Input(id='type', value='MKT', type='text'),
    dcc.Input(id='price', value=400, type='number'),
    # Submit button for the trade
    html.Button('Confirm', id='submit-trade', n_clicks=0),
])


# Callback for what to do when submit-button is pressed
@app.callback(
    [  # there's more than one output here, so you have to use square brackets to pass it in as an array.
        dash.dependencies.Output('info', 'children'),
        dash.dependencies.Output('yield-curve', 'figure'),
        dash.dependencies.Output('blotter', 'data'),
        dash.dependencies.Output('blotter', 'columns'),
        dash.dependencies.Output('actn', 'value'),
        dash.dependencies.Output('type', 'value'),
        dash.dependencies.Output('price', 'value')
    ],
    dash.dependencies.Input('submit-button', 'n_clicks'),
    dash.dependencies.State('start-date', 'date'),
    dash.dependencies.State('end-date', 'date'),
    dash.dependencies.State('starting-cash', 'value'),
    dash.dependencies.State('size', 'value'),
    dash.dependencies.State('md', 'value'),
    prevent_initial_call=True
)
def update_yield_curve(n_clicks, start, end, starting_cash, value, md):
    if md != 'Logistic':
        model_data, test_data = strategy(value, md)
        df, test = backtest1(model_data, test_data)
        if test['Response'][-1] > 0.5:
            ac = 'BUY'
            tp = 'MKT'
        else:
            ac = 'SELL'
            tp = 'LMT'

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df['Date'],
                    y=df['Rate of Return(%)']
                )
            ]
        )
        fig.update_layout(title='Back-Test-Return', yaxis={'hoverformat': '.2%'})
        data = df.dropna(axis=0)
        close = test['Close'][-1]
    else:
        # 应该运行下行语句获取历史数据，但是目前为测试阶段，为了不频繁向bbg发送信息，从之前下载的测试数据集直接读取
        # '''df = req_hisdata('20200101', '20200331', 150)'''
        # 获取测试用历史数据集
        # df = pd.read_csv('test_histdata.csv')

        # 获取回测结果
        if start is not None:
            start = date.fromisoformat(start)
        if end is not None:
            end = date.fromisoformat(end)

        # run the backtest
        test, blotter = backtest2(starting_cash, logistic_strategy, start, end, train_window=100, maxPoints=2000)

        line1 = go.Scatter(
                    x=test['date'],
                    y=test['acc_return'],
                    name='strategy cumulative return',
                    mode='lines+markers'
        )
        line2 = go.Scatter(
                    x=test['date'],
                    y=test['ivv_acc_return'],
                    name='IVV cumulative return',
                    mode='lines+markers'
        )
        fig = go.Figure(data=[line1, line2])
        fig.update_layout(title='Back-Test-Return', yaxis={'hoverformat': '.2%'})

        blotter = blotter.to_dict('records')
        blotter_columns = [
            dict(id='Created', name='Created'),
            dict(id='Action', name='Action'),
            dict(id='Size', name='Size'),
            dict(id='Symbol', name='Symb'),
            dict(
                id='Order Price', name='Order Price', type='numeric',
                format=FormatTemplate.money(2)
            ),
            dict(id='Type', name='Type'),
            dict(id='Filled / Cancelled', name='Filled/Cancelled'),
            dict(id='Filled Price', name='Filled Price',type='numeric',
                format=FormatTemplate.money(2))
        ]


        # 下面三个参数暂时随便写了一下，保证程序运行
        ac = 'BUY'
        tp = 'MKT'
        close = 0

    #     data = data.drop(labels='Rate of Return(%)', axis=1, inplace=True)
    max_rows = 800
    # Return your updated text to currency-output, and the figure to candlestick-graph outputs
    return ('Successfully trained model with window size ' + str(value), fig, blotter,blotter_columns, ac, tp, close)

@app.callback(
    dash.dependencies.Output('info2', 'children'),
    dash.dependencies.Input('submit-trade', 'n_clicks'),
    dash.dependencies.State('actn', 'value'),
    dash.dependencies.State('price', 'value'),
    prevent_initial_call=True
)
def trade(n_clicks, ac, tp, pc):  # Still don't use n_clicks, but we need the dependency
    msg = 1 # 这个也是先随便写了下保证运行
    trade_order = {'action': ac, 'trade_amt': 100, 'trade_currency': 'IVV', 'price': pc}
    # Dump trade_order as a pickle object to a file connection opened with write-in-binary ("wb") permission:
    with open('trade_order.p', "wb") as f:
        pickle.dump(trade_order, f)
    return msg


# Run it!
if __name__ == '__main__':
    app.run_server(debug=False)