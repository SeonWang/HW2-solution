import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable, FormatTemplate
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State
import pandas as pd
from os import listdir, remove
import pickle
from time import sleep
from strategy import *
from backtest import *
from strategyData import *
import matplotlib.pyplot as plt
from sklearn import linear_model
from statistics import *

app = dash.Dash(__name__)

# Define the layout.
app.layout = html.Div([
    # Section title
    html.H1("Section 1: Backtest"),
    html.Div([
        html.H2('Logistic Strategy'),
        html.P('This app explores a simple strategy that works as follows:'),
        html.Ol([
            html.Li([
                "After the market is closed, retrieve the past 100 days (not including today)' " + \
                "data of the following 4 variables:",
                html.Ul([
                    html.Li("ROE: daily IVV ROE"),
                    html.Li("Turnover Ratio: average turnover ratio of the 500 component stocks in S&P 500"),
                    html.Li("Expected Inflation: 10Y Treasury Yield - 10Y TIPS Yield"),
                    html.Li("Trend Score in past 3 days: When one day's return is (> 0, < 0, 0), "
                            "the day get (+1, -1, 0) score. For example, in the past 3 days"
                            ", if the daily return(close / last day's close) is (0.01, 0, -0.01), then "
                            "the scores are (1, 0, -1), the Trend Score = 1 + 0 - 1 = 0"),
                ]),
                "Then, retrieve the past 99 days and today's IVV return(close / last day's close). If the IVV return"
                " is > 0, we denote it as +1 else -1. And we call it IVV signal."
            ]),
            html.Li([
                'Use these 100 data to train a Logistic regression model. The training data are',
                html.Ul([
                    html.Li('y: IVV signal in past 99 days and today.'),
                    html.Li('x: ROE, Turnover Ratio, Expected Inflation and Trend Score in past 100 days')
                ])
            ]),
            html.Li(
                "In step 2, we can get the coefficients of the Logistic regression.\n Then, we put today's "
                "ROE, Turnover Ratio, Expected Inflation and Trend Score into the model. And next day's IVV signal"
            ),
            html.Li([
                "If the IVV signal in next day is +1:",
                html.Ul([
                    html.Li("Use half of the cash to bid a limited price on today's close price"),
                    html.Li("Use half of the cash to bid a market price on next day's open price")
                ]),
                "If the IVV signal in next day is -1:",
                html.Ul([
                    html.Li("Use half of the stock shares to ask a limited price on today's close price"),
                    html.Li("Use half of the stock shares to ask a market price on next day's open price")
                ])]
            )
        ])
    ],
        style={'display': 'inline-block', 'width': '50%'}
    ),
    html.Div([
        html.H2('Data Note & Disclaimer'),
        html.P(
            'This Dash app makes use of Bloomberg\'s Python API to append ' + \
            'the latest historical data!'
        ),
        html.H2('Parameters'),
        html.Ol([
            html.Li(
                "Start Date: The start date of the back test."
            ),
            html.Li(
                "End Date: The end date of the back test. I suggest not to choose today."
            ),
            html.Li(
                "Risk free rate: The default is T-bill's yield."
            )
        ]),
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
        html.Table(
            [html.Tr([
                html.Th('Start Date'),
                html.Th('End Date'),
                html.Th('Bloomberg Identifier'),
                html.Th('Starting Cash'),
                html.Th('Risk Free Rate (%)'),
                html.Th('')
            ])]+
            [html.Tr([
                html.Td(
                    dcc.DatePickerSingle(
                        id='start-date',
                        min_date_allowed=date(2000, 1, 1),
                        max_date_allowed=date.today(),
                        initial_visible_month=date.today(),
                        date=date.today() + timedelta(days=-1),
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
                html.Td(
                    dcc.Input(
                        id='risk-free', type="number",
                        value=0.06,
                        style={'text-align': 'center', 'line-height': '40px', 'font-size': '20px'}
                    )
                ),
                html.Button('Run Backtest', id='submit-button', n_clicks=0,
                            style={'text-align': 'center', 'line-height': '40px', 'font-size': '20px'})
            ])]
        )]
    ),
    # Line break
    html.Br(),
    html.Div(
        html.Table(
            [html.Tr([
                html.Th('Alpha'), html.Th('Beta'),
                html.Th('Annualized GMMR'),
                html.Th('Annualized Volatility'), html.Th('Annualized Sharpe')
                ])] + [html.Tr([
                    html.Td(html.Div(id='strategy-alpha', style={'text-align': 'center'})),
                    html.Td(html.Div(id='strategy-beta', style={'text-align': 'center'})),
                    html.Td(html.Div(id='strategy-gmrr', style={'text-align': 'center'})),
                    html.Td(html.Div(id='strategy-vol', style={'text-align': 'center'})),
                    html.Td(html.Div(id='strategy-sharpe', style={'text-align': 'center'}))
                ])],
            className='main-summary-table',
            style={'display': 'block', 'text-align': 'center'}
        ),
        style={'display': 'block', 'text-align': 'center'}
    ),
    # Div to hold the initial instructions and the updated info once submit is pressed
    html.Div(id='info', children='Enter a your preferred window size and model then press "submit".'),
    html.Div([
        # cumulative return:
        dcc.Graph(id='yield-curve'),
        dcc.Graph(id='daily-return'),
        html.Div([
            html.H2(
                'Trade Blotter',
                style={
                    'display': 'inline-block', 'width': '55%',
                    'text-align': 'center'
                }
            ),
            html.H2(
                'Calender Ledger',
                style={
                    'display': 'inline-block', 'width': '40%',
                    'text-align': 'center'
                }
            )
        ]),
        html.Div(
            DataTable(
                id='blotter',
                fixed_rows={'headers': True},
                style_cell={'textAlign': 'center'},
                style_table={'height': '300px', 'overflowY': 'auto'}
            ),
            style={'display': 'inline-block', 'width': '55%'}
        ),
        html.Div(
            DataTable(
                id='calender',
                fixed_rows={'headers': True},
                style_cell={'textAlign': 'center'},
                style_table={'height': '300px', 'overflowY': 'auto'}
            ),
            style={'display': 'inline-block', 'width': '40%'}
        ),
        dcc.Graph(id='ols')
    ]),
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
        dash.dependencies.Output('daily-return', 'figure'),
        dash.dependencies.Output('ols', 'figure'),
        dash.dependencies.Output('blotter', 'data'),
        dash.dependencies.Output('blotter', 'columns'),
        dash.dependencies.Output('calender', 'data'),
        dash.dependencies.Output('calender', 'columns'),
        dash.dependencies.Output('actn', 'value'),
        dash.dependencies.Output('type', 'value'),
        dash.dependencies.Output('price', 'value'),
        dash.dependencies.Output('strategy-alpha', 'children'),
        dash.dependencies.Output('strategy-beta', 'children'),
        dash.dependencies.Output('strategy-gmrr', 'children'),
        dash.dependencies.Output('strategy-vol', 'children'),
        dash.dependencies.Output('strategy-sharpe', 'children')
    ],
    dash.dependencies.Input('submit-button', 'n_clicks'),
    dash.dependencies.State('start-date', 'date'),
    dash.dependencies.State('end-date', 'date'),
    dash.dependencies.State('starting-cash', 'value'),
    dash.dependencies.State('risk-free', 'value'),
    dash.dependencies.State('size', 'value'),
    dash.dependencies.State('md', 'value'),
    prevent_initial_call=True
)
def update_yield_curve(n_clicks, start, end, starting_cash, risk_free, value, md):
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
        if start is not None:
            start = date.fromisoformat(start)
        if end is not None:
            end = date.fromisoformat(end)
        # run the backtest
        test, blotter, calender = backtest2(starting_cash, logistic_strategy, start, end, train_window=100, maxPoints=2000)

        # figure1
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
        fig1 = go.Figure(data=[line1, line2])
        fig1.update_layout(title='Back-Test-Cumulative-Return', yaxis={'hoverformat': '.2%'})

        # figure2
        line1 = go.Bar(
            x=test['date'],
            y=test['daily_return'],
            name='strategy return'
        )
        line2 = go.Bar(
            x=test['date'],
            y=test['ivv_daily_return'],
            name='IVV return'
        )
        fig2 = go.Figure(data=[line1, line2])
        fig2.update_layout(title='Back-Test-Daily-Return', yaxis={'hoverformat': '.2%'})

        # blotter
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

        # calender
        calender = calender.to_dict('records')
        calender_columns = [
            dict(id='date', name='Date'),
            dict(
                id='close', name='Stock Close', type='numeric',
                format=FormatTemplate.money(2)
            ),
            dict(
                id='cash', name='Cash', type='numeric',
                format=FormatTemplate.money(2)
            ),
            dict(
                id='ivv_value', name='Stock Value', type='numeric',
                format=FormatTemplate.money(2)
            ),
            dict(
                id='total_value', name='Total Value', type='numeric',
                format=FormatTemplate.money(2))
        ]

        # gmrr, vol and shapre ratio
        gmrr = 253 * ((test.acc_return[len(test.index) - 1] + 1 ) ** (1 / len(test.index)) - 1)
        gmrr_str = str(round(gmrr * 100, 3)) + "%"

        vol = (253 ** 0.5) * stdev(test.daily_return)
        vol_str = str(round(vol * 100, 3)) + "%"
        rf = risk_free / 100
        sharpe = round((gmrr - rf) / vol, 3)

        # alpha & beta
        X = test['ivv_daily_return'].values.reshape(-1, 1)
        linreg_model = linear_model.LinearRegression()
        linreg_model.fit(X, test['daily_return'])
        alpha = str(round(linreg_model.intercept_ * 100, 3)) + "%"
        beta = round(linreg_model.coef_[0], 3)
        x_range = np.linspace(X.min(), X.max(), 100)
        y_range = linreg_model.predict(x_range.reshape(-1, 1))
        fig3 = px.scatter(
            test,
            title="Performance against Benchmark",
            x='ivv_daily_return',
            y='daily_return'
        )
        fig3.add_traces(go.Scatter(x=x_range, y=y_range, name='OLS Fit'))

        # no meaning
        ac = 'BUY'
        tp = 'MKT'
        close = 0

    #     data = data.drop(labels='Rate of Return(%)', axis=1, inplace=True)
    max_rows = 800
    # Return your updated text to currency-output, and the figure to candlestick-graph outputs
    return ('Successfully trained model ', fig1, fig2, fig3,
            blotter,blotter_columns, calender, calender_columns, ac, tp, close,
            alpha, beta, gmrr_str, vol_str, sharpe)

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