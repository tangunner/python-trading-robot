import os
import sys

currentdir = f"{str(os.getcwd())}\\python-trading-robot"
parentdir = os.path.dirname(currentdir)
if currentdir not in sys.path:
    sys.path.append(currentdir)
if parentdir not in sys.path:
    sys.path.append(parentdir)
if f"{currentdir}\\pyrobot" not in sys.path:
    sys.path.append(f"{currentdir}\\pyrobot")
if f"{parentdir}\\td-ameritrade-python-api" not in sys.path:
    sys.path.append(f"{parentdir}\\td-ameritrade-python-api")
    sys.path.append(f"{parentdir}\\td-ameritrade-python-api\\td")

import time as time_lib
import pprint
import pathlib
import operator
import pandas as pd
import logging

from datetime import datetime
from datetime import timedelta
from configparser import ConfigParser

from pyrobot.robot import PyRobot
from pyrobot.indicators import Indicators


logging.basicConfig(filename='backtest.log', encoding='utf-8', level=logging.DEBUG)

# Grab configuration values.
config = ConfigParser()
config.read('config/config.ini')

CLIENT_ID = config.get('main', 'CLIENT_ID')
REDIRECT_URI = config.get('main', 'REDIRECT_URI')
CREDENTIALS_PATH = config.get('main', 'JSON_PATH')
ACCOUNT_NUMBER = config.get('main', 'ACCOUNT_NUMBER')

symbol = 'SPY'
paper_trading = True

# Initalize the robot.
trading_robot = PyRobot(
    client_id=CLIENT_ID,
    redirect_uri=REDIRECT_URI,
    credentials_path=CREDENTIALS_PATH,
    trading_account=ACCOUNT_NUMBER,
    paper_trading=paper_trading
)

# Create a Portfolio
trading_robot.create_portfolio()

# trading_robot.portfolio.add_position(
#     symbol=symbol,
#     quantity=10,
#     purchase_price=10,
#     asset_type='equity',
#     purchase_date='2020-04-01'
# )

trading_robot_accounts = trading_robot.session.get_accounts(account=ACCOUNT_NUMBER)
account_cash = trading_robot_accounts['securitiesAccount']['currentBalances']['cashAvailableForTrading']


class PaperTradingAccount(object):
    def __init__(self):
        self.cash = 100000
        self.lotSizing = 10000
        self.minCash = None
        self.minCashThreshold = 0.20
        self.useRealAccountCashBal = False        

account = PaperTradingAccount()

historical_prices = trading_robot.grab_historical_prices(
    period_type='year',
    period=2,
    frequency=1,
    frequency_type='daily',
    symbols=[symbol]
)

# Convert data to a StockFrame. hist_prices['aggregated'] -> List[dicts]
stock_frame = trading_robot.create_stock_frame(data=historical_prices['aggregated'])
trading_robot.portfolio.historical_prices = historical_prices
trading_robot.portfolio.stock_frame = stock_frame

# Create an Indicator object and add an indicator
indicator_client = Indicators(price_data_frame=stock_frame)
indicator_client.st_decline(period=30)
indicator_client._frame.dropna(subset=['st_decline'], inplace=True)

# pd.set_option('display.max_rows', indicator_client._frame.shape[0]+1)
# print(indicator_client._frame)

indicator_client.set_indicator_signal(
    indicator='st_decline',
    buy=0.9,
    sell=1.00,
    condition_buy=operator.le,
    condition_sell=operator.ge,
    buy_max=0.80,
    condition_buy_max=operator.le
)



# Create a new Trade Object.
enter_spy = trading_robot.create_trade(
    trade_id='long_spy',
    enter_or_exit='enter',
    long_or_short='long',
    order_type='mkt'
)

# exit_spy = trading_robot.create_trade(
#     trade_id='exit_long_spy',
#     enter_or_exit='exit',
#     long_or_short='long',
#     order_type='mkt'
# )

# Define a trading dictionary.
trades_dict = {
    symbol: {
        'buy': {
            'trade_func': trading_robot.trades['long_spy'],
            'trade_id': trading_robot.trades['long_spy'].trade_id
        },
        'sell': {
            # 'trade_func': trading_robot.trades['exit_long_spy'],
            # 'trade_id': trading_robot.trades['exit_long_spy'].trade_id
        }
    }
}

for idx, bar in indicator_client._frame.iterrows():

    signals = indicator_client.check_current_signals(bar)

    if not signals['buys'].empty:
        price = bar['close']
        
        # if not account.lotSizing:
        #     account.lotSizing = account.cash * 0.10
        
        quantity=int(account.lotSizing / price)
        gross_value = quantity * price
        
        if trading_robot.portfolio.in_portfolio(symbol):
            projected_val = trading_robot.portfolio.projected_market_value(current_prices={symbol:{'lastPrice':price}})
            account.minCash = (projected_val['total']['total_market_value'] + account.cash) * account.minCashThreshold
            if account.cash - gross_value < account.minCash:
                quantity=int((account.cash - account.minCash) / price)
                gross_value = quantity * price
        
        account.cash -= gross_value
        
        # Add an Order Leg.
        enter_spy.instrument(
            symbol=symbol,
            quantity=quantity,
            asset_type='EQUITY'
        )

        dt = idx[1]
        trading_robot.portfolio.add_position(symbol, 'equity', dt, quantity, price)
        logging.info(f'BUY: {symbol} >> {quantity} @ ${price} >> {dt}')

    # elif not signals['sells'].empty and trading_robot.portfolio.in_portfolio(symbol):
    #     price = bar['close']
    #     quantity=int(trading_robot.portfolio.positions[symbol]['quantity'] * 0.75)
    #     gross_value = quantity * price
    #     account.cash += gross_value
        
    #     # Add an Order Leg.
    #     exit_spy.instrument(
    #         symbol=symbol,
    #         quantity=quantity,
    #         asset_type='EQUITY'
    #     )

    #     trading_robot.portfolio.positions[symbol]['quantity'] -= quantity
    #     dt = idx[1]
    #     logging.info(f'SELL: {symbol} >> {quantity} @ ${price} >> {dt}')

    # Execute Trades.
    trading_robot.execute_signals(
        signals=signals,
        trades_to_execute=trades_dict
    )

    # # Grab the last bar.
    # last_bar_timestamp = trading_robot.stock_frame.frame.tail(
    #     n=1
    # ).index.get_level_values(1)

    # # Wait till the next bar.
    # trading_robot.wait_till_next_bar(last_bar_timestamp=last_bar_timestamp)
