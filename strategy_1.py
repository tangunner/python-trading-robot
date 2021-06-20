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

# # Define mutliple positions to add.
# multi_position = [
#     {
#         'asset_type': 'equity',
#         'quantity': 2,
#         'purchase_price': 4.00,
#         'symbol': 'TSLA',
#         'purchase_date': '2020-01-31'
#     },
#     {
#         'asset_type': 'equity',
#         'quantity': 2,
#         'purchase_price': 4.00,
#         'symbol': 'SQ',
#         'purchase_date': '2020-01-31'
#     }
# ]

# # Grab the New positions
# new_positions = trading_robot.portfolio.add_positions(positions=multi_position)
# pprint.pprint(new_positions)

# # adding a placeholder SPY position that'll be used to get hist prices
# trading_robot_portfolio.add_position(
#     symbol='SPY',
#     asset_type='equity'
# )

# # Add a single position
# trading_robot_portfolio.add_position(
#     symbol='SPY',
#     quantity=10,
#     purchase_price=128,
#     asset_type='equity',
#     purchase_date='2006-01-01'
# )

# # Add another single position
# trading_robot_portfolio.add_position(
#     symbol='AAPL',
#     quantity=10,
#     purchase_price=10,
#     asset_type='equity',
#     purchase_date='2020-04-01'
# )

# # If the Market is open, print some quotes.
# if trading_robot.regular_market_open:
#     pprint.pprint(trading_robot.grab_current_quotes())

# # If the Post Market is Open, do something.
# elif trading_robot.post_market_open:
#     pprint.pprint(trading_robot.grab_current_quotes())

# # If the Pre Market is Open, do something.
# elif trading_robot.pre_market_open:
#     pprint.pprint(trading_robot.grab_current_quotes())

# # Print the Positions
# pprint.pprint(trading_robot_portfolio.positions)

# # Grab the current quotes, for all of our positions.
# current_quotes = trading_robot.grab_current_quotes()

# # Print the Quotes.
# pprint.pprint(current_quotes)

# # Let's see if our Microsoft Position is profitable.
# is_msft_porfitable = trading_robot.portfolio.is_profitable(
#     symbol="MSFT",
#     current_price=current_quotes['MSFT']['lastPrice']
# )
# print("Is Microsoft Profitable: {answer}".format(answer=is_msft_porfitable))

# # Let's get the projected Market Value.
# portfolio_summary = trading_robot.portfolio.projected_market_value(
#     current_prices=current_quotes
# )
# pprint.pprint(portfolio_summary)

# # Create a new Trade Object.
# new_trade = trading_robot.create_trade(
#     trade_id='long_msft',
#     enter_or_exit='enter',
#     long_or_short='short',
#     order_type='lmt',
#     price=150.00
# )

# # Make it Good Till Cancel.
# new_trade.good_till_cancel(cancel_time=datetime.now())

# # Change the session
# new_trade.modify_session(session='am')

# # Add an Order Leg.
# new_trade.instrument(
#     symbol='MSFT',
#     quantity=2,
#     asset_type='EQUITY'
# )

# # Add a Stop Loss Order with the Main Order.
# new_trade.add_stop_loss(
#     stop_size=.10,
#     percentage=False
# )

# # Print out the order.
# pprint.pprint(new_trade.order)

# Grab historical prices, first define the start date and end date.
end_date = datetime.today()
start_date = datetime(2006,1,1)
# start_date = end_date - timedelta(days=30)

# Grab the historical prices.
historical_prices = trading_robot.grab_historical_prices(
    start=start_date,
    end=end_date,
    bar_size=1,
    bar_type='week',
    symbols=['SPY']
)

# historical_prices = trading_robot.grab_historical_prices(
#     start=start_date,
#     end=end_date,
#     bar_size=1,
#     bar_type='minute'
# )

# Convert data to a Data Frame. hist_prices['aggregated'] = a list of dicts,
# with each dict containing price/vol/high/low/etc. stock data
stock_frame = trading_robot.create_stock_frame(
    data=historical_prices['aggregated']
)

# We can also add the stock frame to the Portfolio object.
trading_robot.portfolio.stock_frame = stock_frame

# Additionally the historical prices can be set as well.
trading_robot.portfolio.historical_prices = historical_prices

# # Portfolio Variance
# pprint.pprint(trading_robot.portfolio.portfolio_metrics())

# Create an indicator Object.
indicator_client = Indicators(price_data_frame=stock_frame)

indicator_client.st_decline(period=30)

indicator_client.set_indicator_signal(
    indicator='period_max',
    buy=0.7,
    sell=1.01,
    condition_buy=operator.le,
    condition_sell=operator.ge
)



# # Add the RSI Indicator.
# indicator_client.rsi(period=14)

# # Add the 200 day simple moving average.
# indicator_client.sma(period=200)

# # Add the 50 day simple moving average.
# indicator_client.sma(period=50)

# # Add the 50 day exponentials moving average.
# indicator_client.ema(period=50)

# # Add a signal to check for.
# indicator_client.set_indicator_signal(
#     indicator='rsi',
#     buy=40.0,
#     sell=20.0,
#     condition_buy=operator.ge,
#     condition_sell=operator.le
# )

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
    quantity=10,
    asset_type='EQUITY'
)

# Define a trading dictionary.
trades_dict = {
    'SPY': {
        'trade_func': trading_robot.trades['long_spy'],
        'trade_id': trading_robot.trades['long_spy'].trade_id
    }
}

while True:

    # Grab the latest bar.
    latest_bars = trading_robot.get_latest_bar()

    # Add to the Stock Frame.
    stock_frame.add_rows(data=latest_bars)

    # Refresh the Indicators so the new row is shown in the data.
    indicator_client.refresh()

    print("="*50)
    print("Current StockFrame")
    print("-"*50)
    print(stock_frame.symbol_groups.tail())
    print("-"*50)
    print("")

    # Check for signals.
    signals = indicator_client.check_signals()

    if not signals['buys'].empty:
        trades_dict

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
