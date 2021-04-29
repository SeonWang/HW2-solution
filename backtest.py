from datetime import *
from strategy import *
import pandas as pd
from strategyData import *
from statistics import *
import time

def shares_to_buy(last_day_cash, price):
    if last_day_cash // price < 100:
        actn_shares = 0
    else:
        actn_shares = (last_day_cash // (price * 100)) * 100
    return actn_shares

# def date_to_locate_training(date, train_window):

def backtest1(model_data,test):
    filename = 'finalized_model.sav'
    loaded_model = load(open(filename, 'rb'))
    test_data = model_data[-31:]
    Result = loaded_model.predict(test_data.loc[:, test_data.columns != 'Response'])
    amt = 0
    ror = []
    blotter = pd.DataFrame()
    ledger = pd.DataFrame()
    blotter['Date'] = test_data.index[:]
    blotter['ID'] = test_data['Close'][:]
    blotter['Type'] = test_data['Close'][:]
    blotter['actn'] = test_data['Close'][:]
    blotter['Price'] = test_data['Close'][:]
    blotter['size'] = test_data['Close'][:]
    blotter['symb'] = test_data['Close'][:]
    ledger['Date'] = test_data.index[:]
    ledger['position'] = test_data['Close'][:]
    ledger['Cash'] = test_data['Close'][:]
    ledger['Stock Value'] = test_data['Close'][:]
    ledger['Total Value'] = test_data['Close'][:]
    ledger['Revenue'] = test_data['Close'][:]
    ledger['IVV Yield'] = test_data['IVV'][:]
    ledger['position'][0] = 0
    ledger['Cash'][0] = 1000000
    ledger['Total Value'][0] = 1000000
    ledger['Stock Value'][0] = 1000000
    ledger['Revenue'][0] = 0
    ledger['IVV Yield'][0] = test_data['IVV'][0]
    count = 1
    last = 1000000
    gmrr = 1
    revenue = 0.0
    for i in range(1, 31):
        blotter['ID'][i-1] = count
        ledger['IVV Yield'][i] = test_data['IVV'][i]
        if Result[i - 1] > 0.5:
            blotter['Type'][i-1] = 'MKT'
            blotter['actn'][i-1] = 'BUY'
            blotter['symb'][i-1] = 'IVV'
            blotter['size'][i-1] = 200
            blotter['Price'][i-1] = test_data['Open'][i].round(2)
            ledger['position'][i] = ledger['position'][i-1] + 200
            ledger['Stock Value'][i] = test_data['Close'][i]*ledger['position'][i]
            ledger['Cash'][i] = ledger['Cash'][i-1] - test_data['Open'][i]*200
            ledger['Total Value'][i] = ledger['Stock Value'][i]+ledger['Cash'][i]
            ledger['Revenue'][i] = ledger['Total Value'][i]/ledger['Total Value'][i-1]-1
            gmrr = gmrr *(1+ledger['Revenue'][i])
        else:
            blotter['Type'][i-1] = 'LMT'
            blotter['actn'][i-1] = 'SELL'
            blotter['symb'][i-1] = 'IVV'
            blotter['size'][i-1] = ledger['position'][i-1]
            blotter['Price'][i-1] = test_data['Close'][i-1].round(2)
            ledger['position'][i] = 0
            ledger['Stock Value'][i] = 0
            ledger['Cash'][i] = ledger['Cash'][i-1] + test_data['Close'][i-1]*ledger['position'][i-1]
            ledger['Total Value'][i] = ledger['Stock Value'][i]+ledger['Cash'][i]
            ledger['Revenue'][i] = ledger['Total Value'][i]/ledger['Total Value'][i-1]-1
            gmrr = gmrr *(1+revenue)
        count = count +1
#     test['Response'] =  loaded_model.predict(test.loc[:, test.columns != 'Response'])
    test_data[:]['actn'] = 'BUY'
    vol = np.std(ledger['Revenue'])
    gmrr = pow(gmrr,1/30)-1
    sharp = (gmrr-0.0007)/vol
    return blotter[:-1], ledger, test, sharp

def backtest2(initial_value, strategy, start, end, risk_free, train_window, maxPoints):
    hist_data = req_hisdata(start, end, train_window, maxPoints)

    # if run out of bbg limits, use the histdata
    # hist_data = pd.read_csv('histdata.csv', index_col=0)

    hist_data.index = pd.to_datetime(hist_data.index, format='%Y-%m-%d')
    df = strategy(hist_data, start, train_window)
    initial_open = df['OPEN'].iloc[0]
    account = [[(df.index[0] + timedelta(days=-1)).strftime('%Y-%m-%d'), 0, 0, 0, initial_value, initial_value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    for i in range(0, len(df)):
        data = df[i: i + 1]
        last_day_total_value = account[i][4]
        last_day_cash = account[i][5]
        last_day_ivv_shares = account[i][6]

        date = df[i: i + 1].index[0].strftime('%Y-%m-%d')
        close_1d = data['CLOSE_1D'].iloc[0]
        open = data['OPEN'].iloc[0]
        close = data['CLOSE'].iloc[0]
        actn = data['actn'].iloc[0]

        last_day_cash1 = 0.5 * last_day_cash
        last_day_cash2 = 0.5 * last_day_cash

        if actn == -1:
            market_order_shares = - last_day_ivv_shares // 200 * 100
            limit_order_shares = - (last_day_ivv_shares + market_order_shares)
            actn_shares1 = market_order_shares
            cash_for_actn_shares1 = - open * actn_shares1
            if close_1d >= min(open, close) and close_1d <= max(open, close):
                actn_shares2 = limit_order_shares
                limit_order_status = 'Filled'
            else:
                actn_shares2 = 0
                limit_order_status = 'Cancelled'
            cash_for_actn_shares2 = - close_1d *actn_shares2
        else:
            market_order_shares = shares_to_buy(last_day_cash1, open)
            limit_order_shares = shares_to_buy(last_day_cash2, close_1d)
            actn_shares1 = market_order_shares
            cash_for_actn_shares1 = - open * actn_shares1
            if close_1d >= min(open, close) and close_1d <= max(open, close):
                actn_shares2 = limit_order_shares
                limit_order_status = 'Filled'
            else:
                actn_shares2 = 0
                limit_order_status = 'Cancelled'
            cash_for_actn_shares2 = - close_1d *actn_shares2

        actn_shares = actn_shares1 + actn_shares2
        cash = last_day_cash + cash_for_actn_shares1 + cash_for_actn_shares2
        ivv_shares = last_day_ivv_shares + actn_shares
        ivv_value = ivv_shares * close
        total_value = cash + ivv_value
        daily_return = total_value / last_day_total_value - 1
        acc_return = total_value / initial_value - 1

        ivv_daily_return = close / close_1d - 1
        ivv_acc_return = close / initial_open - 1
        account_data = [date, close_1d, open, close, total_value, cash,
                        ivv_shares, ivv_value, actn,
                        market_order_shares, limit_order_shares, limit_order_status,
                        daily_return, acc_return, ivv_daily_return, ivv_acc_return]
        account.append(account_data)
    account = pd.DataFrame(account,
                           columns=['date', 'close_1d', 'open', 'close', 'total_value', 'cash',
                                    'ivv_shares', 'ivv_value', 'actn',
                                    'market_order_shares', 'limit_order_shares', 'limit_order_status',
                                    'daily_return', 'acc_return', 'ivv_daily_return', 'ivv_acc_return'])
    account.to_csv('backtest_results.csv')

    trade_blotter = to_trade_blotter(account)
    calender_ledger = to_calender_ledger(account)
    daily_return = to_daily_return(account)
    monthly_return, monthly_expected_return, monthly_volatility, monthly_sharpe_ratio = to_monthly_return(account, risk_free)
    return account, trade_blotter, calender_ledger, daily_return, monthly_return,monthly_expected_return, monthly_volatility, monthly_sharpe_ratio

def to_trade_blotter(backtest_results):
    backtest_results['actn'] = backtest_results['actn'].apply(lambda x: 'BUY' if x == 1 else 'SELL')
    backtest_results['market_order_shares'] = backtest_results['market_order_shares'].apply(
        lambda x: x if x > 0 else -x)
    backtest_results['limit_order_shares'] = backtest_results['limit_order_shares'].apply(lambda x: x if x > 0 else -x)

    market_orders = backtest_results[['date', 'actn', 'market_order_shares', 'open']]
    limit_orders = backtest_results[['date', 'actn', 'limit_order_shares', 'close_1d', 'limit_order_status']]

    market_orders = market_orders[market_orders['market_order_shares'] != 0]
    limit_orders = limit_orders[limit_orders['limit_order_shares'] != 0]

    market_orders.columns = ['Created', 'Action', 'Size', 'Order Price']
    market_orders[['Type', 'Filled / Cancelled']] = ['MKT', 'Filled']

    limit_orders.columns = ['Created', 'Action', 'Size', 'Order Price', 'Filled / Cancelled']
    limit_orders['Type'] = 'LMT'

    trade_blotter = pd.concat([market_orders, limit_orders], axis=0)

    trade_blotter = trade_blotter.sort_index(ascending=False). \
        groupby(by='Created').apply(lambda x: x.sort_values('Type', ascending=False))
    trade_blotter['Symbol'] = 'IVV'
    trade_blotter = trade_blotter.reindex(columns=['Created', 'Action', 'Size',
                                                   'Symbol', 'Order Price', 'Type', 'Filled / Cancelled'])
    trade_blotter['Filled Price'] = trade_blotter.apply(
        lambda x: x['Order Price'] if x['Filled / Cancelled'] == 'Filled' else None, axis=1)
    return trade_blotter

def to_calender_ledger(backtest_results):
    calender_ledger = backtest_results[['date', 'close', 'cash', 'ivv_value', 'total_value']]
    return calender_ledger

def to_daily_return(backtest_results):
    daily_return = backtest_results[['date', 'daily_return', 'acc_return', 'ivv_daily_return', 'ivv_acc_return']]
    return daily_return

def to_monthly_return(backtest_results, risk_free):
    monthly_return = backtest_results[['date', 'total_value', 'close']]
    initial_value = monthly_return.iloc[0, 1]
    initial_close = monthly_return.iloc[1, 2]

    monthly_return['date2'] = pd.to_datetime(monthly_return['date'], format='%Y-%m-%d')
    monthly_return = monthly_return.groupby(pd.Grouper(key='date2', freq='m')).last()
    monthly_return.columns = ['mon_date', 'mon_total_value', 'mon_close']
    first_month_value = monthly_return.iloc[0, 1]
    first_month_close = monthly_return.iloc[0, 2]
    first_month_return = first_month_value / initial_value - 1
    first_month_ivv_return = first_month_close / initial_close - 1

    monthly_return['value_delta'] = monthly_return['mon_total_value'].diff()
    monthly_return['value_previous'] = monthly_return['mon_total_value'].shift(periods=1)
    monthly_return['close_delta'] = monthly_return['mon_close'].diff()
    monthly_return['close_previous'] = monthly_return['mon_close'].shift(periods=1)
    monthly_return['strategy_return'] = monthly_return['value_delta'] / monthly_return['value_previous']
    monthly_return['ivv_return'] = monthly_return['close_delta'] / monthly_return['close_previous']
    monthly_return.iloc[0, 7] = first_month_return
    monthly_return.iloc[0, 8] = first_month_ivv_return

    monthly_expected_return = monthly_return['strategy_return'].mean()
    monthly_volatility = monthly_return['strategy_return'].std()
    monthly_sharpe_ratio = (monthly_expected_return - risk_free / 100) / monthly_volatility
    monthly_expected_return_str = str(round(monthly_expected_return * 100, 3)) + '%'
    monthly_volatility_str = str(round(monthly_volatility * 100, 3)) + '%'
    monthly_sharpe_ratio_str = str(round(monthly_sharpe_ratio, 3))

    return monthly_return, monthly_expected_return_str, monthly_volatility_str, monthly_sharpe_ratio_str