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
from pprint import pprint

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
benchmark = 'SPY'
paper_trading = True
backtesting = True
debug_timing = True

start = datetime.now().timestamp() if debug_timing else 0

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
        self.minReturn = 0.20
        self.priceSlippage = 0.01
        self.txCosts = 0.00
        self.saleSignalRatio = 0.75
        self.backtestEndDate = datetime.today()
        self.backtestStartDate = datetime.today() - timedelta(365)

        self.STCapGainTax = 0.20                  # ST tax = personal tax rate
        self.LTCapGainTax = 0.15                  # LT tax typically 15%

    def getUpdatedMinCash(self, trading_robot):
        """Updates the min cash balance"""
        
        if not trading_robot.portfolio or not trading_robot.portfolio.positions:
            self.minCash = self.cash * self.minCashThreshold
            return self.minCash

        portfolio_mv = trading_robot.portfolio.market_value
        if not trading_robot.portfolio.market_value:
            current_prices = {}
            latest_prices = trading_robot.get_latest_bar()
            
            for symbol in trading_robot.portfolio.positions:
                current_prices[symbol] = [bar for bar in latest_prices if bar['symbol'] == symbol][0]
                
            portfolio_mv = trading_robot.portfolio.projected_market_value(
                current_prices)['total']['market_value']

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

# add the 'sell' indicators
premium = 1.05
indicator_client.sma_premium(period=30, premium=premium)

# add the 'buy' indicators
discount_ratios = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]   # 1 - pct discount
interval = 0.10
indicator_client.discount_ratio(period=30, discounts=discount_ratios, interval=interval)

# remove rows with nan values
indicator_client._frame.dropna(inplace=True)

# Add the close price/avg price premium (sell) signal
indicator_client.set_indicator_signal(
    indicator=f'sma_prem_{str(int(premium*100))}',
    buy=1.01,                      # outside possible vals to restrict buying
    sell=1,
    condition_buy=operator.eq,
    condition_sell=operator.ge
)

# Add each of the discount (buy) signals
for discount in discount_ratios:
    col = f'max_disc_{str(int(discount*100))}'
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

exit_spy = trading_robot.create_trade(
    trade_id='exit_long_spy',
    enter_or_exit='exit',
    long_or_short='long',
    order_type='mkt',
    tax_lot_method='specific_lot'
)

# create the trade dict
trades_dict = {
    symbol: {
        'buy': {
            'trade_func': trading_robot.trades['long_spy'],
            'trade_id': trading_robot.trades['long_spy'].trade_id
        },
        'sell': {
            'trade_func': trading_robot.trades['exit_long_spy'],
            'trade_id': trading_robot.trades['exit_long_spy'].trade_id
        }
    }
}


# # display the indicators with signals
# new_frame = indicator_client._frame.iloc[:, [1, -12, -11, -8, -7, -6, -5]]
# new_frame2 = new_frame[new_frame.isin([1]).any(axis=1)]
# pd.set_option('display.max_rows', new_frame2.shape[0]+1)
# print(new_frame2)
# print(' ')
    
print(f'Elapsed time: {datetime.now().timestamp() - start} secs \n ') if start else print(' ')
start = datetime.now().timestamp() if debug_timing else 0

for idx, bar in indicator_client._frame.groupby(level=1):
    date = bar.index[0][1]
    current_prices = 0
    
    # update and store the portfolio market values
    if trading_robot.portfolio.positions:
        current_prices = {}
        for symbol in trading_robot.portfolio.positions:
            current_prices[symbol] = {}
            current_prices[symbol]['close'] = bar['close'].item()
    
    trading_robot.portfolio.update_metrics(
        current_prices=current_prices, 
        date=date, 
        cash=account.cash
        )

    # grab the indicators that are in-use for shares already bought or sold
    locked_indicators = trading_robot.portfolio.check_portfolio_indicators(
        symbol=bar.index[0][0])

    signals = indicator_client.check_current_signals(
        bar=bar, 
        locked_indicators=locked_indicators
        )

    skip_sells_signals = True
    skip_buys_signals = True

    # check first for sell signals to free up capital, in case needed to buy
    if signals['sells'].notnull().any():
        if not signals['sells'].empty and trading_robot.portfolio.get_ownership_status(symbol):
            skip_sells_signals = False
            print('--'*50)
            print('Sell Signals:')

            logging.info(f'{date} -- Locked indicators: {locked_indicators}')
            print(f'{date} -- Locked indicators: {locked_indicators}')

            # grab the first (highest cost) lot purchased
            first_lot = trading_robot.portfolio.position_lots[symbol][0]

            # skip signal if returns don't hurdle
            if first_lot['annualized_return'] < account.minReturn or \
                first_lot['total_return'] < account.minReturn:
                skip_sells_signals = True
                logging.info(f'{date} -- {symbol} Skipped Sell Flag: returns < {account.minReturn}')
                print(f'{date} -- {symbol} Skipped Sell Flag: returns < {account.minReturn}')

            else:
                price = bar['close'].item()
                sma_premium = bar['close_to_sma'].item()
                total_quantity = int(trading_robot.portfolio.positions[symbol]['quantity'])
                sale_quantity = round(account.saleSignalRatio * total_quantity, 0)
                
                # For papertrading, adj sale price to allow for slippage
                if paper_trading:
                    slippage = account.priceSlippage * price
                    price = price - slippage
                
                sale_market_value = sale_quantity * price
                account.cash += sale_market_value
                account.getUpdatedMinCash(trading_robot)

                # Add an order leg
                exit_spy.instrument(
                    symbol=symbol,
                    quantity=sale_quantity,
                    asset_type='EQUITY'
                )

                trading_robot.portfolio.reduce_position(
                    symbol=symbol,
                    quantity=sale_quantity,
                    tax_lot_method='fifo',
                    indicator_used=signals['sells'].name
                )

                tot_shrs = trading_robot.portfolio.positions[symbol]['quantity']
                print(f'{date} -- SELL: {symbol} >> {sale_quantity} @ ${price}')
                print(f'SMA Premium: {sma_premium} | Total Shares: {tot_shrs} | Avail. Cash: ${account.cash}')
                print()
                logging.info(f'{date} -- SELL: {symbol} >> {quantity} @ ${price}')
                logging.info(f'SMA Premium: {sma_premium} | Total Shares: {tot_shrs} | Avail. Cash: ${account.cash}')
        

    if signals['buys'].notnull().any():
        if not signals['buys'].empty:
            skip_buys_signals = False
            print('--'*50)
            print('Buy Signals:')

            logging.info(f'{date} -- Locked indicators: {locked_indicators}')
            print(f'{date} -- Locked indicators: {locked_indicators}')
            
            """NOTE: a.item() only takes the first val in the arr; wouldn't work if
            there were multiple symbols passed"""

            price = bar['close'].item()
            disc_ratio = bar['discount_ratio'].item()
            quantity = int(account.lotSizing / price)
            cost_basis = price * quantity

            # adj quantity if cost dips into min cash balance
            if ((account.cash - cost_basis) < account.minCash):
                quantity = int((account.cash - account.minCash) / price)
            
            # do nothing if we can't afford more shares
            if not quantity:
                logging.info(f'{date} -- {symbol} Skipped Buy Flag (discount_ratio = {disc_ratio}): cash = ${account.cash}')
                skip_buys_signals = True
            
            else:
                if paper_trading:
                    slippage = account.priceSlippage * price
                    price = price + slippage

                cost_basis = price * quantity
                account.cash -= cost_basis
                account.getUpdatedMinCash(trading_robot)
                
                # Add order leg
                enter_spy.instrument(
                    symbol=symbol,
                    quantity=quantity,
                    asset_type='EQUITY'
                )

                """NOTE: This may need to be moved to after execute_signals for the actual
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
                print()
                logging.info(f'{date} -- BUY: {symbol} >> {quantity} @ ${price}')
                logging.info(f'Discount Ratio: {disc_ratio} | Total Shares: {tot_shrs} | Avail. Cash: ${account.cash}')

    if skip_sells_signals and skip_buys_signals:
        continue

    # Execute Trades
    trading_robot.execute_signals(
        signals=signals,
        trades_to_execute=trades_dict
    )
    
    logging.info(f'Signals executed.')


print(f'Elapsed time: {datetime.now().timestamp() - start} secs \n ') if start else print(' ')
start = datetime.now().timestamp() if debug_timing else 0


"""
----------------------------------------------
Calculate Strategy & Benchmark Metrics/Returns
----------------------------------------------

"""


if trading_robot.regular_market_open:
    current_prices = trading_robot.grab_current_quotes()
    date = datetime.today()

# if market closed, use last close price as current price
else:
    current_prices = {}
    
    try:
        for symbol in trading_robot.portfolio.positions:
            current_prices[symbol] = {}
            current_prices[symbol]['close'] = trading_robot.portfolio.positions[symbol]['current_price']
    
    except KeyError:
        last_prices = trading_robot.get_latest_bar()
        for symbol in trading_robot.portfolio.positions:
            current_prices[symbol] = {}
            current_prices[symbol]['close'] = [
                p['close'] for p in last_prices if p['symbol'] == symbol][0]

# add current prices to the port tracker and calc port max drawdown
trading_robot.portfolio.update_metrics(
    current_prices=current_prices, 
    date=date, 
    cash=account.cash
    )
portfolio_max_drawdown = trading_robot.portfolio.get_portfolio_max_drawdown(window=252)

# grab historical prices for the benchmark and calc its max drawdown
benchmark_historical_prices = trading_robot.grab_historical_prices(
    period_type='year',
    period=2,
    frequency=1,
    frequency_type='daily',
    symbols=[benchmark]
)
benchmark_max_drawdown = trading_robot.get_benchmark_max_drawdown(
    prices_data=benchmark_historical_prices['aggregated'], window=252)

print(f'portfolio_max_drawdown: {portfolio_max_drawdown}')
print(f'benchmark_max_drawdown: {benchmark_max_drawdown}')

print(f'Elapsed time: {datetime.now().timestamp() - start} secs \n ') if start else print(' ')

logging.info('--'*50)
logging.info(f'STRATEGY RETURNS')
logging.info(f'Market Value: {trading_robot.portfolio.market_value}')
logging.info(f'Invested Capital: {trading_robot.portfolio.invested_capital}')
logging.info(f'Profit Loss: {trading_robot.portfolio.profit_loss}')
logging.info(f'Max Drawdown: {portfolio_max_drawdown}')
logging.info(f'Total Return: {trading_robot.portfolio.total_return}')
ann_return = round(trading_robot.portfolio.total_return * (365 / timedelta(milliseconds=(account.backtestEndDate - account.backtestStartDate)).days), 4)
logging.info(f'Annualized Return: {ann_return}')
logging.info(' ')

logging.info(f'BENCHMARK ({symbol}) RETURNS')
spy_tot_return = round(((historical_prices['aggregated'][-1]['close'] - historical_prices['aggregated'][0]['close']) / historical_prices['aggregated'][0]['close']), 4)
spy_ann_return = round(spy_tot_return * (365 / timedelta(milliseconds=(account.backtestEndDate - account.backtestStartDate)).days), 4)
logging.info(f'Max Drawdown: {benchmark_max_drawdown}')
logging.info(f'Total Return: {spy_tot_return}')
logging.info(f'Annualized Return: {spy_ann_return}')
logging.info(' ')
logging.info(f'Strategy Alpha: {(ann_return - spy_ann_return)}')
logging.info('--'*50)



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
