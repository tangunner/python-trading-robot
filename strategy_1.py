import os
import sys

# print(sys.path)
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
# print(sys.path)

import time as time_lib
import pprint
import pathlib
import operator
import pandas as pd

from datetime import datetime
from datetime import timedelta
from configparser import ConfigParser

from pyrobot.robot import PyRobot
from pyrobot.indicators import Indicators

# Grab configuration values.
config = ConfigParser()
config.read('config/config.ini')

CLIENT_ID = config.get('main', 'CLIENT_ID')
REDIRECT_URI = config.get('main', 'REDIRECT_URI')
CREDENTIALS_PATH = config.get('main', 'JSON_PATH')
ACCOUNT_NUMBER = config.get('main', 'ACCOUNT_NUMBER')

# ------------------------- #
### PAPER TRADING EXAMPLE ###
# ------------------------- #

# Initalize the robot.
trading_robot = PyRobot(
    client_id=CLIENT_ID,
    redirect_uri=REDIRECT_URI,
    credentials_path=CREDENTIALS_PATH,
    trading_account=ACCOUNT_NUMBER,
    paper_trading=True
)

# Create a Portfolio
trading_robot_portfolio = trading_robot.create_portfolio()

trading_robot_accounts = trading_robot.session.get_accounts(
    account=ACCOUNT_NUMBER
)

# Cash available in my actual account (not paper trading account)
account_cash = trading_robot_accounts['securitiesAccount']['currentBalances']['cashAvailableForTrading']

# Adjust the balance if no cash in the account
account_cash = account_cash if account_cash != 0.0 else 200000.0

# # Grab historical prices, first define the start date and end date.
# end_date = datetime.now()
# start_date = end_date - timedelta(days=30)

# # Grab the historical prices.
# historical_prices = trading_robot.grab_historical_prices(
#     start=start_date,
#     end=end_date,
#     bar_size=1,
#     bar_type='minute',
#     symbols=['SPY']
# )

historical_prices = trading_robot.grab_historical_prices(
    period_type='year',
    period=2,
    frequency=1,
    frequency_type='daily',
    symbols=['SPY']
)

# Convert data to a Data Frame. hist_prices['aggregated'] = a list of dicts,
# with each dict containing price/vol/high/low/etc. stock data
stock_frame = trading_robot.create_stock_frame(
    data=historical_prices['aggregated']
)

# We can also add the stock frame to the Portfolio object.
trading_robot.portfolio.stock_frame = stock_frame
trading_robot.portfolio.historical_prices = historical_prices

# Create an indicator object and add an indicator
indicator_client = Indicators(price_data_frame=stock_frame)
frame = indicator_client.st_decline(period=30)

# # display the data
# pd.set_option('display.max_rows', indicator_client._frame.shape[0]+1)
# print(frame)

indicator_client.set_indicator_signal(
    indicator='st_decline_10',
    buy=0.9,
    sell=1.01,
    condition_buy=operator.le,
    condition_sell=operator.ge,
    buy_max=0.85,
    condition_buy_max=operator.le
)

indicator_client.set_indicator_signal(
    indicator='st_decline_20',
    buy=0.8,
    sell=1.01,
    condition_buy=operator.le,
    condition_sell=operator.ge,
    buy_max=0.75,
    condition_buy_max=operator.le
)

indicator_client.set_indicator_signal(
    indicator='st_decline_30',
    buy=0.7,
    sell=1.01,
    condition_buy=operator.le,
    condition_sell=operator.ge,
    buy_max=0.65,
    condition_buy_max=operator.le
)

indicator_client.set_indicator_signal(
    indicator='st_decline_40',
    buy=0.6,
    sell=1.01,
    condition_buy=operator.le,
    condition_sell=operator.ge,
    buy_max=0.55,
    condition_buy_max=operator.le
)

indicator_client.set_indicator_signal(
    indicator='st_decline_50',
    buy=0.5,
    sell=1.01,
    condition_buy=operator.le,
    condition_sell=operator.ge,
    buy_max=0.45,
    condition_buy_max=operator.le
)

indicator_client.set_indicator_signal(
    indicator='st_decline_60',
    buy=0.4,
    sell=1.01,
    condition_buy=operator.le,
    condition_sell=operator.ge,
    buy_max=0.35,
    condition_buy_max=operator.le
)


# Create a new Trade Object.
new_trade = trading_robot.create_trade(
    trade_id='long_spy',
    enter_or_exit='enter',
    long_or_short='long',
    order_type='mkt'
)

# Add an Order Leg.
new_trade.instrument(
    symbol='SPY',
    quantity=1,
    asset_type='EQUITY'
)

# # Define a trading dictionary.
# trades_dict = {
#     'SPY': {
#         'trade_func': trading_robot.trades['long_spy'],
#         'trade_id': trading_robot.trades['long_spy'].trade_id
#     }
# }

# Define a trading dictionary.
trades_dict = {
    'SPY': {
        'buy': {
            'trade_func': trading_robot.trades['long_spy'],
            'trade_id': trading_robot.trades['long_spy'].trade_id
        },
        'sell': {

        }
    }
}

try:
    positions = trading_robot.get_positions(account_number=ACCOUNT_NUMBER)
except:
    positions = trading_robot_portfolio.positions
print(positions)


"""Below is for doing live trading. NOTE THIS STILL NEEDS TO INCLUDE
FUNCTIONALITY TO FLAG WHEN A TRADE HAS BEEN EXECUTED - OTHERWISE IT WILL KEEP
EXECUTING THE SAME TRADES"""

# Continues checking for indicators as long as the market is open
while trading_robot.regular_market_open:

    # Grab the latest bar.
    latest_bars = trading_robot.get_latest_bar()

    # Add to the Stock Frame.
    stock_frame.add_rows(data=latest_bars)

    # Refresh the Indicators so the new row is shown in the data.
    indicator_client.refresh()

    print("="*50)
    print("Current StockFrame:")
    print("-"*50)
    print(stock_frame.symbol_groups.tail())
    print("-"*50)
    print("")

    # Check for signals.
    signals = indicator_client.check_signals()

    # Execute Trades.
    trading_robot.execute_signals(
        signals=signals,
        trades_to_execute=trades_dict
    )

    # Grab the last bar.
    last_bar_timestamp = trading_robot.stock_frame.frame.tail(
        n=1
    ).index.get_level_values(1)

    # Wait till the next bar.
    trading_robot.wait_till_next_bar(last_bar_timestamp=last_bar_timestamp)
