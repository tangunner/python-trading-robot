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

import operator
import numpy as np
import pandas as pd
import logging

from datetime import datetime, timedelta
from configparser import ConfigParser

from pyrobot.robot import PyRobot
from pyrobot.indicators import Indicators


logging.basicConfig(filename='backtest.log', encoding='utf-8', level=logging.DEBUG)

config = ConfigParser()
config.read('config/config.ini')

CLIENT_ID = config.get('main', 'CLIENT_ID')
REDIRECT_URI = config.get('main', 'REDIRECT_URI')
CREDENTIALS_PATH = config.get('main', 'JSON_PATH')
ACCOUNT_NUMBER = config.get('main', 'ACCOUNT_NUMBER')

symbol = 'SPY'
paper_trading = True
backtesting = True

trading_robot = PyRobot(
    client_id=CLIENT_ID,
    redirect_uri=REDIRECT_URI,
    credentials_path=CREDENTIALS_PATH,
    trading_account=ACCOUNT_NUMBER,
    paper_trading=paper_trading
)

trading_robot.create_portfolio()
trading_robot_accounts = trading_robot.get_accounts()


class PaperTradingAccount(object):
    def __init__(self):
        self.cash = 100000
        self.lotSizing = 10000
        self.useRealCashBal = False
        self.minCashThreshold = 0.20
        self.minCash = self.cash * self.minCashThreshold
        self.txCosts = 0.01
        self.STCapGainTax = 0.20                  # ST tax = personal tax rate
        self.LTCapGainTax = 0.15                  # LT tax typically 15%
        self.backtestEndDate = datetime.today()
        self.backtestStartDate = datetime.today() - timedelta(365)
            
    def getUpdatedMinCash(self, trading_robot):
        if not trading_robot.portfolio or not trading_robot.portfolio.positions:
            self.minCash = self.cash * self.minCashThreshold
            return self.minCash
        
        if trading_robot.regular_market_open():
            current_prices=trading_robot.grab_current_quotes()
        else:
            current_prices = {}
            for symbol in trading_robot.portfolio.positions:
                current_prices[symbol] = [bar for bar in trading_robot.get_latest_bar() if bar['symbol'] == symbol][0]
            
        portfolio_mv = trading_robot.portfolio.projected_market_value(current_prices)['total']['market_value']
        self.minCash = (portfolio_mv + self.cash) * self.minCashThreshold
        return self.minCash

    def setBacktestDates(self, historical_prices):
        self.backtestStartDate = historical_prices['aggregated'][0]['datetime']
        self.backtestEndDate = historical_prices['aggregated'][-1]['datetime']

    def setRealCashBalance(self, trading_robot_accounts):
        account_cash = trading_robot_accounts['securitiesAccount']['currentBalances']['cashAvailableForTrading']
        if account_cash:
            self.cash = account_cash


historical_prices = trading_robot.grab_historical_prices(
    period_type='year',
    period=2,
    frequency=1,
    frequency_type='daily',
    symbols=[symbol]
)

account = PaperTradingAccount()
account.setBacktestDates(historical_prices)

# Convert data to a StockFrame
stock_frame = trading_robot.create_stock_frame(data=historical_prices['aggregated'])
trading_robot.portfolio.historical_prices = historical_prices
trading_robot.portfolio.stock_frame = stock_frame

# Create an Indicator object and add the indicators
indicator_client = Indicators(price_data_frame=stock_frame)
indicator_client.change_in_price()

# premium = 1.20
# indicator_client.close_to_avg_ratio(period=30, premium=premium)

# discount_ratios = np.linspace(0.5, 1.0, num=5, endpoint=False)
discount_ratios = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]   # 1 - pct discount
interval = 0.10
indicator_client.discount_ratio(period=30, discounts=discount_ratios, interval=interval)

# # Add the close price/avg price premium (sell) signal
# indicator_client.set_indicator_signal(
#     indicator=f'close_to_avg_{str(int(premium*100))}',
#     buy=1.01,                      # outside possible vals to restrict buying
#     sell=1,
#     condition_buy=operator.eq,
#     condition_sell=operator.ge
# )

# Add each of the discount (buy) signals
for discount in discount_ratios:
    col = f'discount_ratio_{str(int(discount*100))}'
    indicator_client.set_indicator_signal(
        indicator=col,
        buy=1,
        sell=1.01,
        condition_buy=operator.eq,
        condition_sell=operator.eq
    )

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

# pd.set_option('display.max_rows', indicator_client._frame.shape[0]+1)
# print(indicator_client._frame)

account.getUpdatedMinCash(trading_robot)


# iter over all the dates in the historical data
for idx, bar in indicator_client._frame.groupby(level=1):
    date = bar.index[0][1]

    # update and store the portfolio market values
    if trading_robot.portfolio.positions:
        current_prices = {}
        for symbol in trading_robot.portfolio.positions:
            current_prices[symbol] = historical_prices[symbol]['candles'][-1]

    # grab the indicators that were used to buy or sell current shares
    locked_indicators = trading_robot.portfolio.check_portfolio_indicators(symbol=bar.index[0][0])

    if locked_indicators:
        signals = indicator_client.check_current_signals(bar=bar, locked_indicators=locked_indicators)
    else:
        signals = indicator_client.check_current_signals(bar=bar)

    # # check first for sell signals to free up capital to buy, in case it's needed
    # if not signals['sells'].empty and trading_robot.portfolio.get_ownership_status(symbol):
    #     print('--'*50)
    #     print(signals['sells'])

    #     # date = bar.index[0][1]
    #     # price = bar['close']
    #     # disc_ratio = bar['discount_ratio']
    #     # quantity = int(account.lotSizing / price)
    #     # cost_basis = quantity * price
        
    #     date = bar.index[0][1]
    #     price = bar['close']
    #     quantity = int(trading_robot.portfolio.positions[symbol]['quantity'] * 0.75)
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
    

    # check if buy signals
    if not signals['buys'].empty:
        print('--'*50)
        print('Buy Signals:')
        print(signals['buys'])
        print()

        logging.info(f'{date} -- Locked indicators: {locked_indicators}')
        print(f'{date} -- Locked indicators: {locked_indicators}')
        
        """NOTE: a.item() only takes the first val in the arr; wouldn't work if
        there were multiple symbols passed"""

        price = bar['close'].item()
        disc_ratio = bar['discount_ratio'].item()
        quantity = int(account.lotSizing / price)
        cost_basis = quantity * price

        # if any securities in the port, update min cash using MV + cash
        if trading_robot.portfolio.positions:
            
            """NOTE: a.any() updates the quantity/cost basis if any of the vals
            in the arr are < minCash"""
            # if ((account.cash - cost_basis) < account.minCash).any():

            # adj quantity if cost dips into min cash balance
            if ((account.cash - cost_basis) < account.minCash):
                quantity = int((account.cash - account.minCash) / price)
                cost_basis = quantity * price
        
        # do nothing if we can't afford more shares
        if not quantity:
            logging.info(f'{date} -- {symbol} Skipped Buy Flag (discount_ratio = {disc_ratio}): cash = ${account.cash}')
            continue
        
        account.cash -= cost_basis
        account.cash -= account.txCosts*cost_basis
        print(f'tx_costs: {account.txCosts*cost_basis}')
        
        # Add order leg
        enter_spy.instrument(
            symbol=symbol,
            quantity=quantity,
            asset_type='EQUITY'
        )

        """NOTE: This needs to be moved to after execute_signals for the actual
        program; should only add the position to the portfolio once the trade
        response indicates the status is executed"""

        # Add the new position to the portfolio
        trading_robot.portfolio.add_position(
            symbol=symbol, 
            asset_type='equity', 
            purchase_date=date, 
            quantity=quantity, 
            purchase_price=price,
            indicator_used=signals['buys'].name
        )
        
        tot_shrs = trading_robot.portfolio.positions[symbol]['quantity']
        print(f'{date} -- BUY: {symbol} >> {quantity} @ ${price}')
        print(f'Discount Ratio: {disc_ratio} | Total Shares: {tot_shrs} | Avail. Cash: ${account.cash}')
        logging.info(f'{date} -- BUY: {symbol} >> {quantity} @ ${price}')
        logging.info(f'Discount Ratio: {disc_ratio} | Total Shares: {tot_shrs} | Avail. Cash: ${account.cash}')

    else:
        # logging.info(f'{bar.index[0][1]} -- No signals')
        print(f'{bar.index[0][1]} -- No signals')
        continue

    # Execute Trades.
    trading_robot.execute_signals(
        signals=signals,
        trades_to_execute=trades_dict
    )
    
    logging.info(f'Signals executed.')

# # logging.info(f'Loop completed. (Time for full loop: {datetime.now() - beg_loop})')


"""
Calculate Strategy's Returns
"""

if not trading_robot.portfolio.positions:
    current_mv = account.cash

market_open_flag = trading_robot.regular_market_open

# if market closed, use last close price as current price
if market_open_flag:
    current_prices = trading_robot.grab_current_quotes()
else:
    current_prices = {}
    try:
        for symbol in trading_robot.portfolio.positions:
            current_prices[symbol] = trading_robot.portfolio.positions[symbol]['current_price']
    except KeyError:
        for symbol in trading_robot.portfolio.positions:
            current_prices[symbol] = historical_prices[symbol]['candles'][-1]

proj_val = trading_robot.portfolio.projected_market_value(current_prices)
port_value = proj_val['total']['market_value']
inv_cap = proj_val['total']['invested_capital']

print(f'SUMMARY RETURNS')
print('--'*25)
print('Market Value:', trading_robot.portfolio.market_value)
print('Invested Capital:', trading_robot.portfolio.invested_capital)
print('Profit Loss:', trading_robot.portfolio.profit_loss)
print('Max Drawdown:')
print('Total Return:', trading_robot.portfolio.total_return)
ann_return = round((trading_robot.portfolio.total_return * (365 / (account.backtestEndDate - account.backtestStartDate).days)), 4)
print('Annualized Return:', ann_return)
print()
print(f'BENCHMARK ({symbol}) RETURNS')
spy_tot_return = round(((historical_prices['aggregated'][-1]['close'] - historical_prices['aggregated'][0]['close']) / historical_prices['aggregated'][0]['close']), 4)
spy_ann_return = round((spy_tot_return * (365 / (account.backtestEndDate - account.backtestStartDate).days)), 4)
print('Max Drawdown:')
print('Total Return:', spy_tot_return)
print('Annualized Return:', spy_ann_return)
print()
print('Strategy Alpha:', (ann_return - spy_ann_return))
print('--'*40)


# port_return_incl_cash = round(((port_value + account.cash - 100000) / (100000)), 4)
# port_return_excl_cash = round(((port_value - inv_cap) / inv_cap), 4)
# print('port_return_incl_cash', port_return_incl_cash)
# print('port_return_excl_cash', port_return_excl_cash)
# print('--'*40)

# logging.info('--'*25)
# logging.info(f'Portfolio return (excl cash): {port_return_excl_cash}')
# logging.info(f'Portfolio return (incl cash): {port_return_incl_cash}')
# logging.info('--'*50)

# while True:

#     # Grab the latest bar.
#     latest_bars = trading_robot.get_latest_bar()

#     # Add to the Stock Frame.
#     stock_frame.add_rows(data=latest_bars)

#     # Refresh the Indicators so the new row is shown in the data.
#     indicator_client.refresh()

#     print("="*50)
#     print("Current StockFrame")
#     print("-"*50)
#     print(stock_frame.symbol_groups.tail())
#     print("-"*50)
#     print("")

#     # Check for signals.
#     signals = indicator_client.check_signals()

#     # Execute Trades.
#     trading_robot.execute_signals(
#         signals=signals,
#         trades_to_execute=trades_dict
#     )

#     # Grab the last bar.
#     last_bar_timestamp = trading_robot.stock_frame.frame.tail(
#         n=1
#     ).index.get_level_values(1)

#     # Wait till the next bar.
#     trading_robot.wait_till_next_bar(last_bar_timestamp=last_bar_timestamp)
