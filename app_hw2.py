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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define the layout.
app.layout = html.Div([
    # Section title
    html.H1("Section 1: Backtest"),
    html.Div([
        html.H2('1.1 Logistic Strategy'),
        html.P('This app explores a simple strategy that works as follows:'),
        html.Ol([
            html.Li([
                "After the market is closed, retrieve the past 100 days (not including today)' " + \
                "data of the following 5 variables:",
                html.Ul([
                    html.Li("IVV signal: retrieve the past 99 days and today's IVV return(close / last day's close - 1). "
                        "If the IVV return is >= 0, we denote it as +1, else -1. And we call it IVV signal."),
                    html.Li("ROE: daily IVV ROE"),
                    html.Li("Turnover Ratio: average turnover ratio of the 500 component stocks in S&P 500"),
                    html.Li("Expected Inflation: 10Y Treasury Yield - 10Y TIPS Yield"),
                    html.Li("Trend Score in past 3 days: When one day's return is (> 0, < 0, 0), "
                            "the day get (+1, -1, +1) score. For example, in the past 3 days"
                            ", if the daily return(close / last day's close) is (0.01, 0, -0.01), then "
                            "the scores are (1, 1, -1), the Trend Score = 1 + 1 - 1 = 1")
                ]),
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
                "ROE, Turnover Ratio, Expected Inflation and Trend Score into the model. And get next day's IVV signal"
            ),
            html.Li([
                "If the IVV signal in next day is +1:",
                html.Ul([
                    html.Li("Use half of the cash to bid a limited price on today's close price"),
                    html.Li("Use half of the cash to bid a market price on next day's open price")
                ]),
                "  If the IVV signal in next day is -1:",
                html.Ul([
                    html.Li("Use half of the stock shares to ask a limited price on today's close price"),
                    html.Li("Use half of the stock shares to ask a market price on next day's open price")
                ])]
            )
        ])
    ],
        style={'display': 'inline-block'}
    ),
    html.Div([
        html.H2('1.1.1 Data Note & Disclaimer'),
        html.P(
            'This Dash app makes use of Bloomberg\'s Python API to append ' + \
            'the latest historical data!'
        ),
        html.H2('1.1.2 Parameters'),
        html.Ol([
            html.Li(
                "Start Date: The start date of the back test."
            ),
            html.Li(
                "End Date: The end date of the back test. I suggest not to choose today."
            ),
            html.Li(
                "Risk free rate: The default is T-bill's yield."
            ),
            html.Li(
                "Assume 253 trading days when annualizing the return."
            )
        ]),
        dcc.RadioItems(
            options=[
                {'label': 'Logistic', 'value': 'Logistic'}
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
                        value=0.03,
                        style={'text-align': 'center', 'line-height': '40px', 'font-size': '20px'}
                    )
                ),
                html.Button('Run Backtest', id='submit-button', n_clicks=0,
                            style={'text-align': 'center', 'height': '66px', 'line-height': '40px', 'font-size': '20px'})
            ])]
        )]
    ),
    # Line break
    html.Br(),
    html.Div(
        html.Table(
            [html.Tr([
                html.Th('Alpha'), html.Th('Beta'),
                html.Th('Annualized GMRR'),
                html.Th('Annualized Volatility'),
                html.Th('Annualized Sharpe'),
                html.Th('Monthly Expected Return'),
                html.Th('Monthly Volatility'),
                html.Th('Sharpe (Monthly)')
                ])] + [html.Tr([
                    html.Td(html.Div(id='strategy-alpha', style={'text-align': 'center'})),
                    html.Td(html.Div(id='strategy-beta', style={'text-align': 'center'})),
                    html.Td(html.Div(id='strategy-gmrr', style={'text-align': 'center'})),
                    html.Td(html.Div(id='strategy-vol', style={'text-align': 'center'})),
                    html.Td(html.Div(id='strategy-sharpe', style={'text-align': 'center'})),
                    html.Td(html.Div(id='mon-return', style={'text-align': 'center'})),
                    html.Td(html.Div(id='mon-vol', style={'text-align': 'center'})),
                    html.Td(html.Div(id='mon-sharpe', style={'text-align': 'center'}))
                ])],
            className='main-summary-table'
        )
    ),
    html.Div([
        # cumulative return:
        dcc.Graph(id='yield-curve'),
        dcc.Graph(id='daily-return'),
        html.Div([
            html.H2(
                'Trade Blotter',
                style={
                    'display': 'inline-block', 'width': '50%',
                    'text-align': 'center'
                }
            ),
            html.H2(
                'Calender Ledger',
                style={
                    'display': 'inline-block', 'width': '50%',
                    'text-align': 'center'
                }
            ),
        ]),
        html.Div(
            DataTable(
                id='blotter',
                fixed_rows={'headers': True},
                style_cell={'textAlign': 'center'},
                style_table={'height': '300px', 'overflowY': 'auto'}
            ),
            style={'display': 'inline-block', 'width': '50%'}
        ),
        html.Div(
            DataTable(
                id='calender',
                fixed_rows={'headers': True},
                style_cell={'textAlign': 'center'},
                style_table={'height': '300px', 'overflowY': 'auto'}
            ),
            style={'display': 'inline-block', 'width': '50%'}
        ),
        html.Div([
            html.H2(
                'Daily Return',
                style={
                    'display': 'inline-block','width': '50%',
                    'text-align': 'center'
                }
            ),
            html.H2(
                'Monthly Return',
                style={
                    'display': 'inline-block','width': '50%',
                    'text-align': 'center'
                }
            )

        ]),
        html.Div(
            DataTable(
                id='daily-return-table',
                fixed_rows={'headers': True},
                style_cell={'textAlign': 'center'},
                style_table={'height': '300px', 'overflowY': 'auto'}
            ),
            style={'display': 'inline-block','width': '50%'}
        ),
        html.Div(
            DataTable(
                id='monthly-return-table',
                fixed_rows={'headers': True},
                style_cell={'textAlign': 'center'},
                style_table={'height': '300px', 'overflowY': 'auto'}
            ),
            style={'display': 'inline-block','width': '50%'}
        ),

        dcc.Graph(id='ols')
    ]),


    # Another line break
    html.Br(),
    html.Div([
        html.Div([
            html.H2('1.2: Machine Learning Strategy'),
            html.P('This app explores a simple strategy that works as follows:'),
            html.Ol([
                html.Li([
                    "While the market is closed, retrieve the past N days' " + \
                    "worth of data for:",
                    html.Ul([
                        html.Li("IVV: daily open, close, yield(calculated)"),
                        html.Li(
                            "US Treasury bond yield for 5,10 and 30 years."
                        ),
                        html.Li("Price change of oil(EIA)."),
                    ])
                ]),
                html.Li([
                    'Fit different machine learning models using features listed below to predict' + \
                    " whether the IVV would have yield greater than 0 next day:",
                    html.Ul([
                        html.Li(
                            'the output (y): the yield of IVV next day would be greater than 0(1 for True and 0 for false)'),
                        html.Li(
                            "the input (x): the yield of IVV on previous N days, IVV' moving average, yield of bonds, yield of EIA"),
                        html.Li('the models: KNN, Dicision Tree, Loglinear Classifier')
                    ])
                ]),
                html.Li(
                    "After the model is being trained, we use the model to predict each day's IVV yield," + \
                    'we use the model result of last 30 days to do a back-test. '
                ),
                html.Li(
                    'If the predicted output is 1, which means the IVV would have postive return next day. We submit one trade:'),
                html.Ul([
                    html.Li(
                        'A market order to BUY 100 shares of IVV, which ' + \
                        'fills at open price the next trading day.'
                    )
                ]),
                html.Li(
                    'If the predicted output is 0, which means the IVV would have negative return next day. We submit one trade:'),
                html.Ul([
                    html.Li(
                        'A limit order to SELL all shares of IVV, which ' + \
                        'fills at the close price of the last trading day.'
                    )
                ])
            ])
        ],
            style={'display': 'inline-block'}
        ),
        html.Div([
            html.H2('1.2.1 Data Note & Disclaimer'),
            html.P(
                'This Dash app makes use of yahoo finance data to fit the model ' + \
                "using pandas_datareader package to read yahoo finance's stock and bond data." + \
                'The original data contains close, open, low, high price and we can use them to calculate the yield.' + \
                "These are all the work we done in fetching data and preprocessing it."
            ),
            html.H2('1.2.2 Parameters'),
            html.Ol([
                html.Li(
                    "N: number of days of the moving average of IVV yield, which would be added as a feature into the model"
                ),
                html.Li(
                    "model: Which specific machine learning model would be used in training dataset."
                )
            ]),
            html.H2('Window Size (Moving average of x days.)'),
            html.Br(),
            dcc.Slider(
                id='size',
                min=7, max=100,
                marks={days: str(days) for days in range(1, 101, 3)},
                value=30),
            html.Br(),
            html.H2('Model Selection'),
            dcc.RadioItems(
                options=[
                    {'label': 'Decision Tree', 'value': 'Decision Tree'},
                    {'label': 'Loglinear', 'value': 'Loglinear'},
                    {'label': 'KNN', 'value': 'KNN'}
                ],
                id='md1',
                value='Decision Tree',
                labelStyle={'display': 'inline-block'}
            ),
            html.Br(),
            # Submit button:
            html.Button('Submit', id='submit-button1', n_clicks=0),
            html.Div(id='info', children='Enter a your preferred window size and model then press "submit".'),
        ],
            style={
                'display': 'inline-block'
            }
    ),
    # Line break
    html.Br(),
    html.Div([
        html.H2(
            'Trade Ledger',
            style={
                'display': 'inline-block', 'text-align': 'center',
                'width': '100%'
            }
        ),
        DataTable(
            id='ledger1',
            fixed_rows={'headers': True},
            style_cell={'textAlign': 'center'},
            style_table={'height': '300px', 'overflowY': 'auto'}
        )
    ]),
    html.Div([
        html.H2(
            'Blotters',
            style={
                'display': 'inline-block', 'text-align': 'center',
                'width': '100%'
            }
        ),
        DataTable(
            id='blotters1',
            fixed_rows={'headers': True},
            style_cell={'textAlign': 'center'},
            style_table={'height': '300px', 'overflowY': 'auto'}
        )
    ]),

    html.Div([
        html.H2('Yield-curve'),
        # Candlestick graph goes here:
        dcc.Graph(id='yield-curve1')
    ])]),
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
        dash.dependencies.Output('yield-curve', 'figure'),
        dash.dependencies.Output('daily-return', 'figure'),
        dash.dependencies.Output('ols', 'figure'),
        dash.dependencies.Output('blotter', 'data'),
        dash.dependencies.Output('blotter', 'columns'),
        dash.dependencies.Output('calender', 'data'),
        dash.dependencies.Output('calender', 'columns'),
        dash.dependencies.Output('daily-return-table', 'data'),
        dash.dependencies.Output('daily-return-table', 'columns'),
        dash.dependencies.Output('monthly-return-table', 'data'),
        dash.dependencies.Output('monthly-return-table', 'columns'),
        dash.dependencies.Output('strategy-alpha', 'children'),
        dash.dependencies.Output('strategy-beta', 'children'),
        dash.dependencies.Output('strategy-gmrr', 'children'),
        dash.dependencies.Output('strategy-vol', 'children'),
        dash.dependencies.Output('strategy-sharpe', 'children'),
        dash.dependencies.Output('mon-return', 'children'),
        dash.dependencies.Output('mon-vol', 'children'),
        dash.dependencies.Output('mon-sharpe', 'children')
    ],
    dash.dependencies.Input('submit-button', 'n_clicks'),
    dash.dependencies.State('start-date', 'date'),
    dash.dependencies.State('end-date', 'date'),
    dash.dependencies.State('starting-cash', 'value'),
    dash.dependencies.State('risk-free', 'value'),
    # dash.dependencies.State('size', 'value'),
    dash.dependencies.State('md', 'value'),
    prevent_initial_call=True
)
def update_fig_table(n_clicks, start, end, starting_cash, risk_free, md):
    if start is not None:
        start = date.fromisoformat(start)
    if end is not None:
        end = date.fromisoformat(end)
    # run the backtest
    test, blotter, calender, daily_return, monthly_return, monthly_expected_return, monthly_volatility, \
    monthly_sharpe_ratio = backtest2(starting_cash, logistic_strategy, start, end, risk_free,
                                     train_window=100, maxPoints=2000)
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
    fig1 = go.Figure()
    fig1.add_traces(data=[line1, line2])
    fig1.update_layout(title='Back-Test-Cumulative-Return', yaxis={'hoverformat': '.2%'},
                        xaxis=dict(rangeslider=dict(visible=True), type="date"))

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
    fig2 = go.Figure()
    fig2.add_traces(data=[line1, line2])
    fig2.update_layout(title='Back-Test-Daily-Return', yaxis={'hoverformat': '.2%'},
                        xaxis=dict(rangeslider=dict(visible=True), type="date"))

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
        dict(id='Filled Price', name='Filled Price', type='numeric',
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

    daily_return = daily_return.to_dict('records')
    daily_return_columns = [
        dict(id='date', name='Date'),
        dict(
            id='daily_return', name='Strategy Daily Return', type='numeric',
            format=FormatTemplate.percentage(2)
        ),
        dict(
            id='acc_return', name='Strategy Cumulative Return', type='numeric',
            format=FormatTemplate.percentage(2)
        ),
        dict(
            id='ivv_daily_return', name='IVV Daily Return', type='numeric',
            format=FormatTemplate.percentage(2)
        ),
        dict(
            id='ivv_acc_return', name='IVV Cumulative Return', type='numeric',
            format=FormatTemplate.percentage(2))
    ]


    monthly_return = monthly_return.to_dict('records')
    monthly_return_columns = [
        dict(id='mon_date', name='Date'),
        dict(
            id='strategy_return', name='Strategy Monthly Return', type='numeric',
            format=FormatTemplate.percentage(2)
        ),
        dict(
            id='ivv_return', name='IVV Monthly Return', type='numeric',
            format=FormatTemplate.percentage(2)
        )
    ]

    # gmrr, vol and shapre ratio
    gmrr = 253 * ((test.acc_return[len(test.index) - 1] + 1) ** (1 / len(test.index)) - 1)
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

    #     data = data.drop(labels='Rate of Return(%)', axis=1, inplace=True)
    # Return your updated text to currency-output, and the figure to candlestick-graph outputs
    return (fig1, fig2, fig3,
            blotter, blotter_columns, calender, calender_columns, daily_return, daily_return_columns, monthly_return, monthly_return_columns,
            alpha, beta, gmrr_str, vol_str, sharpe, monthly_expected_return, monthly_volatility, monthly_sharpe_ratio)


@app.callback(
    [  # there's more than one output here, so you have to use square brackets to pass it in as an array.
        dash.dependencies.Output('info', 'children'),
        dash.dependencies.Output('yield-curve1', 'figure'),
        dash.dependencies.Output('blotters1', 'columns'),
        dash.dependencies.Output('blotters1', 'data'),
        dash.dependencies.Output('ledger1', 'columns'),
        dash.dependencies.Output('ledger1', 'data'),
        dash.dependencies.Output('actn', 'value'),
        dash.dependencies.Output('type', 'value'),
        dash.dependencies.Output('price', 'value')
    ],
    dash.dependencies.Input('submit-button1', 'n_clicks'),
    dash.dependencies.State('size', 'value'),
    dash.dependencies.State('md1', 'value'),
    prevent_initial_call=True
)
def update_yield_curve(n_clicks, value, md):
    model_data, test_data = strategy(value, md)

    blotter, ledger, test, sharp = backtest1(model_data, test_data)

    if test['Response'][-1] > 0.5:
        ac = 'BUY'
        tp = 'MKT'
    else:
        ac = 'SELL'
        tp = 'LMT'

    fig = go.Figure(
        data=[
            go.Scatter(
                x=ledger['Date'],
                y=ledger['Revenue'],
                name='Asset Return'),
            go.Scatter(
                x=ledger['Date'],
                y=ledger['IVV Yield'],
                name='IVV Return')
        ]
    )
    fig.update_layout(title='Back-Test-Yield', yaxis={'hoverformat': '.2%'})
    #     data = df.dropna(axis=0)
    blotter.reset_index()
    blotter = blotter.to_dict('records')
    blotter_columns = [
        dict(id='Date', name='Date'),
        dict(id='ID', name='ID'),
        dict(id='Type', name='order type'),
        dict(id='actn', name='Action'),
        dict(
            id='Price', name='Order Price', type='numeric',
            format=FormatTemplate.money(2)
        ),
        dict(id='size', name='Order Amount', type='numeric'
             ),
        dict(id='symb', name='Symb')
    ]
    ledger = ledger.to_dict('records')
    ledger_columns = [
        dict(id='Date', name='Date'),
        dict(id='position', name='position'),
        dict(id='Cash', name='Cash'),
        dict(
            id='Stock Value', name='Stock Value', type='numeric',
            format=FormatTemplate.money(2)
        ),
        dict(
            id='Total Value', name='Total Value', type='numeric',
            format=FormatTemplate.money(2)
        ),
        dict(id='Revenue', name='Revenue', type='numeric',
             format=FormatTemplate.percentage(2)
             ),
        dict(id='IVV Yield', name='IVV Yield', type='numeric',
             format=FormatTemplate.percentage(2)
             )
    ]
    return (
    'Successfully trained model with window size ' + str(value), fig, blotter_columns, blotter, ledger_columns, ledger,
    ac, tp, test['Close'][-1])


@app.callback(
    dash.dependencies.Output('info2', 'children'),
    dash.dependencies.Input('submit-trade', 'n_clicks'),
    dash.dependencies.State('actn', 'value'),
    dash.dependencies.State('price', 'value'),
    prevent_initial_call=True
)
def trade(n_clicks, ac, tp, pc):  # Still don't use n_clicks, but we need the dependency
    msg = ''
    trade_order = {'action': ac, 'trade_amt': 100, 'trade_currency': 'IVV', 'price': pc}
    # Dump trade_order as a pickle object to a file connection opened with write-in-binary ("wb") permission:
    with open('trade_order.p', "wb") as f:
        pickle.dump(trade_order, f)
    return msg

# Run it!
if __name__ == '__main__':
    app.run_server(debug=False)