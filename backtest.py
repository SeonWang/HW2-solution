from datetime import *
from strategy import *
import pandas as pd
from strategyData import *

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
    df = pd.DataFrame()
    df['Date'] = test_data.index[1:]
    df['ID'] = test_data['Close'][1:]
    df['Type'] = test_data['Close'][1:]
    df['actn'] = test_data['Close'][1:]
    df['Price'] = test_data['Close'][1:]
    df['size'] = test_data['Close'][1:]
    df['symb'] = test_data['Close'][1:]
    count = 1
    last = 0.0
    cost = 0
    revenue = 0.0
    for i in range(1, 31):
        df['ID'][i-1] = count
        if Result[i - 1] > 0.5:
            df['Type'][i-1] = 'MKT'
            df['actn'][i-1] = 'BUY'
            df['symb'][i-1] = 'IVV'
            df['size'][i-1] = 100
            df['Price'][i-1] = test_data['Open'][i].round(2)
            amt = amt + 1
            cost = cost + 100*test_data['Open'][i]
            revenue = revenue + amt * 100 * test_data['Close'][i] / cost - 1
            ror.append(revenue.round(4))
            last = revenue
            revenue = 0.0
        else:
            df['Type'][i-1] = 'LMT'
            df['actn'][i-1] = 'SELL'
            df['symb'][i-1] = 'IVV'
            df['size'][i-1] = amt*100
            df['Price'][i-1] = test_data['Close'][i-1].round(2)
            ror.append(round(last,4))
            amt = 0
            revenue = last
            cost = 0.0
        count = count +1
    # test['Response'] =  loaded_model.predict(test.loc[:, test.columns != 'Response'])
    # test_data[:]['actn'] = 'BUY'
    df["Rate of Return(%)"] = ror
    return df, test

def backtest2(initial_value, strategy, start, end, train_window, maxPoints):
    hist_data = req_hisdata(start, end, train_window, maxPoints)
    df = strategy(hist_data, start, train_window)
    initial_open = df['OPEN'].iloc[0]
    account = [[(df.index[0] + timedelta(days=-1)).strftime('%Y-%m-%d'), 0, 0, 0, initial_value, initial_value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    for i in range(0, len(df)):
        data = df[i: i + 1]
        last_day_totoal_value = account[i][4]
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
        daily_return = total_value / last_day_totoal_value - 1
        acc_return = total_value / initial_value - 1

        ivv_daily_return = close / close_1d - 1
        ivv_acc_return = close / initial_open - 1
        account_data = [date, close_1d, open, close, total_value, cash,
                        ivv_shares, ivv_value, actn,
                        market_order_shares, limit_order_shares, limit_order_status,
                        daily_return, acc_return, ivv_daily_return, ivv_acc_return]
        account.append(account_data)
    account = pd.DataFrame(account,
                           columns=['date', 'close_1d', 'open', 'close', 'totoal_value', 'cash',
                                    'ivv_shares', 'ivv_value', 'actn',
                                    'market_order_shares', 'limit_order_shares', 'limit_order_status',
                                    'daily_return', 'acc_return', 'ivv_daily_return', 'ivv_acc_return'])
    account.to_csv('backtest_results.csv')
    trade_blotter = to_trade_blotter(account)
    return account, trade_blotter

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
