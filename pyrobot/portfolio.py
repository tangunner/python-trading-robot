import numpy as np
from datetime import datetime, timedelta

from pandas import DataFrame
from typing import Tuple
from typing import List
from typing import Optional


from pyrobot.stock_frame import StockFrame
from td.client import TDClient


class Portfolio():

    def __init__(self, account_number: Optional[str] = None) -> None:
        """Initalizes a new instance of the Portfolio object.

        Keyword Arguments:
        ----
        account_number {str} -- An accout number to associate with the Portfolio. (default: {None})
        """

        self.cash = 0.00
        self.positions = {}
        self.position_lots = {}
        self.positions_count = 0

        self.total_return = 0.00
        self.profit_loss = 0.00
        self.market_value = 0.00
        self.invested_capital = 0.00
        self.risk_tolerance = 0.00
        self.account_number = account_number

        self._historical_prices = []
        self.performance_tracker = {}

        self._td_client: TDClient = None
        self._stock_frame: StockFrame = None
        self._stock_frame_daily: StockFrame = None

    def add_positions(self, positions: List[dict]) -> dict:
        """Add Multiple positions to the portfolio at once.

        This method will take an iterable containing the values
        normally passed through in the `add_position` endpoint and
        then adds each position to the portfolio.

        Arguments:
        ----
        positions {list[dict]} -- Multiple positions with the required arguments
        to be added.
        position_lots -- 

        Returns:
        ----
        {dict} -- The current positions in the portfolio.

        Usage:
        ----
            >>> # Define mutliple positions to add.
            >>> multi_position = [
                {
                    'asset_type': 'equity',
                    'quantity': 2,
                    'purchase_price': 4.00,
                    'symbol': 'TSLA',
                    'purchase_date': '2020-01-31'
                },
                {
                    'asset_type': 'equity',
                    'quantity': 2,0
                    'purchase_price': 4.00,
                    'symbol': 'SQ',
                    'purchase_date': '2020-01-31'
                }
            ]
            >>> new_positions = trading_robot.portfolio.add_positions(positions=multi_position)
            {
                'SQ': {
                    'asset_type': 'equity',
                    'purchase_date': '2020-01-31',
                    'purchase_price': 4.00,
                    'quantity': 2,
                    'symbol': 'SQ'
                },
                'TSLA': {
                    'asset_type': 'equity',
                    'purchase_date': '2020-01-31',
                    'purchase_price': 4.00,
                    'quantity': 2,
                    'symbol': 'TSLA'
                }
            }
        """

        if isinstance(positions, list):

            # Loop through each position.
            for position in positions:

                # Add the position.
                self.add_position(
                    symbol=position['symbol'],
                    asset_type=position['asset_type'],
                    quantity=position.get('quantity', 0),
                    purchase_price=position.get('purchase_price', 0.0),
                    purchase_date=position.get('purchase_date', None)
                )

            return self.positions

        else:
            raise TypeError('Positions must be a list of dictionaries.')

    def add_position(self, symbol: str, asset_type: str, purchase_date: Optional[str] = None, 
                     quantity: int = 0, purchase_price: float = 0.0, indicator_used: str = None) -> dict:
        """Adds a single new position to the the portfolio.

        Arguments:
        ----
        symbol {str} -- The Symbol of the Financial Instrument. Example: 'AAPL' or '/ES'

        asset_type {str} -- The type of the financial instrument to be added. For example,
            'equity', 'forex', 'option', 'futures'

        Keyword Arguments:
        ----
        quantity {int} -- The number of shares or contracts you own. (default: {0})

        purchase_price {float} -- The price at which the position was purchased. (default: {0.00})

        purchase_date {str} -- The date which the asset was purchased. Must be ISO Format "YYYY-MM-DD"
            For example, "2020-04-01" (default: {None})

        Returns:
        ----
        {dict} -- A dictionary object that represents a position in the portfolio.

        Usage:
        ----
            >>> portfolio = Portfolio()
            >>> new_position = Portfolio.add_position(symbol='MSFT', 
                    asset_type='equity', 
                    quantity=2, 
                    purchase_price=4.00,
                    purchase_date="2020-01-31"
                )
            >>> new_position
            {
                'asset_type': 'equity', 
                'quantity': 2, 
                'purchase_price': 4.00,
                'symbol': 'MSFT',
                'purchase_date': '2020-01-31'
            }
        """

        # if security already owned, update the existing position
        if self.in_portfolio(symbol) and self.positions[symbol]['ownership_status']:

            # add the new lot to the existing position_lots entry
            position_lot = {}
            position_lot['quantity'] = quantity
            position_lot['purchase_price'] = purchase_price
            position_lot['purchase_date'] = purchase_date
            position_lot['cost_basis'] = quantity * purchase_price            
            position_lot['asset_type'] = asset_type
            position_lot['indicator_used'] = indicator_used
            self.position_lots[symbol].append(position_lot)

            # update the position values (purchase price is the WA price)
            self.positions[symbol]['quantity'] += quantity
            self.positions[symbol]['cost_basis'] += quantity * purchase_price
            self.positions[symbol]['purchase_price'] = (
                self.positions[symbol]['cost_basis'] / self.positions[symbol]['quantity']
            )


        else:
            # if security not owned yet, add it to the positions
            self.positions[symbol] = {}
            self.positions[symbol]['symbol'] = symbol
            self.positions[symbol]['quantity'] = quantity
            self.positions[symbol]['purchase_price'] = purchase_price
            self.positions[symbol]['purchase_date'] = purchase_date
            self.positions[symbol]['cost_basis'] = quantity * purchase_price
            self.positions[symbol]['asset_type'] = asset_type


            if purchase_date:
                self.positions[symbol]['ownership_status'] = True
            else:
                self.positions[symbol]['ownership_status'] = False

            # also add the security to the position lots
            self.position_lots[symbol] = []
            position_lot = {}
            position_lot['quantity'] = quantity
            position_lot['purchase_price'] = purchase_price
            position_lot['purchase_date'] = purchase_date
            position_lot['cost_basis'] = quantity * purchase_price
            position_lot['asset_type'] = asset_type
            position_lot['indicator_used'] = indicator_used
            self.position_lots[symbol].append(position_lot)

        return self.positions[symbol]

    def check_portfolio_indicators(self, indicator: str=None, symbol: str=None):
        """Returns the set of all indicators currently in use in all
        position_lots in the portfolio. If indicator given, returns True if it
        has been used to purchase shares in the active portfolio.

        If symbol given and no indicator, returns a set of all indicators
        for the position_lots of that security. Returns True if the indicator is
        given and found in security's position_lots.
        """
        
        if not self.positions and not self.position_lots:
            return None
        
        if not symbol:
            all_indicators = []
            for ticker, lots in self.position_lots.items():
                ticker_indicators = [all_indicators.append(lot['indicator_used']) for lot in lots]
                # all_indicators.append(symbol_indicators)
            indicators = set(all_indicators)
        else:
            indicators = None
            if symbol in self.position_lots.keys():
                indicators = set([lot['indicator_used'] for lot in self.position_lots[symbol]])

        return indicator in indicators if indicator else indicators
    
    def remove_position(self, symbol: str) -> Tuple[bool, str]:
        """Deletes a single position from the portfolio.

        Arguments:
        ----
        symbol {str} -- The symbol of the instrument to be deleted. Example: 'AAPL' or '/ES'

        Returns:
        ----
        {Tuple[bool, str]} -- Returns `True` if successfully deleted, `False` otherwise 
            along with a message.

        Usage:
        ----
            >>> portfolio = Portfolio()

            >>> new_position = Portfolio.add_position(
                    symbol='MSFT', 
                    asset_type='equity', 
                    quantity=2, 
                    purchase_price=4.00,
                    purchase_date="2020-01-31"
                )
            >>> delete_status = Portfolio.delete_position(symbol='MSFT')
            >>> delete_status
            (True, 'MSFT was successfully removed.')

            >>> delete_status = Portfolio.delete_position(symbol='AAPL')
            >>> delete_status
            (False, 'AAPL did not exist in the porfolio.')
        """

        if symbol in self.positions:
            del self.positions[symbol]
            if symbol in self.position_lots:
                del self.position_lots[symbol]
            return (True, "{symbol} was successfully removed.".format(symbol=symbol))
        else:
            return (False, "{symbol} did not exist in the porfolio.".format(symbol=symbol))
    
    def get_performance_tracker(self, totals_only=False):
        """Returns the performance_tracker dict"""
        
        if not self.performance_tracker:
            raise AttributeError('Must create performance tracker before calling.')
        
        tracker = self.performance_tracker
        
        # if we only want the 'total' portfolio metrics
        if totals_only:
            tracker = {}
            
            # reformat dict for making into a DF
            for dt in self.performance_tracker:
                tracker[dt] = {}
                tracker[dt]['market_value'] = self.performance_tracker[dt]['total']['market_value']
                tracker[dt]['invested_capital'] = self.performance_tracker[dt]['total']['invested_capital']
                tracker[dt]['total_profit_or_loss'] = self.performance_tracker[dt]['total']['total_profit_or_loss']
                tracker[dt]['total_return'] = self.performance_tracker[dt]['total']['total_return']
                    
        return tracker
    
    def update_metrics(self, current_prices, date=None):
        """Updates the returns for each position and the overall portfolio and
        stores them in the performance tracker
        
        current_prices {Dict} -- dict of historical prices of the form
        current_prices[symbol]['close'] = <value>
        date {datetime} -- the date corresponding to the current_prices
        """
        
        if not date:
            date = datetime.today()

        # if there are no positions in the portfolio yet
        if current_prices == 0:

            # add placeholders for each 'total' portfolio metric for that date 
            self.performance_tracker[date] = {}
            self.performance_tracker[date]['total'] = {}
            self.performance_tracker[date]['total']['market_value'] = 1
            self.performance_tracker[date]['total']['invested_capital'] = 0
            self.performance_tracker[date]['total']['total_profit_or_loss'] = 0
            self.performance_tracker[date]['total']['total_return'] = 0
            self.performance_tracker[date]['total']['count_profitable_positions'] = 0
            self.performance_tracker[date]['total']['count_non_profitable_positions'] = 0
            self.performance_tracker[date]['total']['count_breakeven_positions'] = 0
            
            # # grab all of the non-null price vals already in the perf tracker
            # prices = [
            #     self.performance_tracker[key] for key in self.performance_tracker if self.performance_tracker[key]
            # ]
            
            # if prices:
            #     for key in prices[0]:
            #         self.performance_tracker[date][key] = 1
            # else:
            #     self.performance_tracker[date]['market_value'] = 1
        
        # # if first call to update_metrics
        # else:
        #     self.performance_tracker[date] = {}
        #     self.performance_tracker[date]['market_value'] = 1
            return
        
        # calc the portfolio values for the timestep
        portfolio_metrics = self.projected_market_value(current_prices, date)
        
        # add the updated values for the timestep to the performance tracker
        self.performance_tracker[date] = portfolio_metrics
        return portfolio_metrics


    def total_allocation(self) -> dict:
        """Returns a summary of the portfolio by asset allocation."""

        total_allocation = {
            'equity': [],
            'fixed_income': [],
            'options': [],
            'futures': [],
            'forex': []
        }

        if len(self.positions.keys()) > 0:
            for symbol in self.positions:
                total_allocation[self.positions[symbol]['asset_type']].append(self.positions[symbol])
            return total_allocation

    def portfolio_variance(self, weights: dict, covariance_matrix: DataFrame) -> dict:

        sorted_keys = list(weights.keys())
        sorted_keys.sort()

        sorted_weights = np.array([weights[symbol] for symbol in sorted_keys])
        portfolio_variance = np.dot(
            sorted_weights.T,
            np.dot(covariance_matrix, sorted_weights)
        )

        return portfolio_variance

    def update_risk_metrics(self) -> dict:
        """Calculates different portfolio risk metrics using daily data.

        Overview:
        ----
        To build an effective summary of our portfolio we will need to
        calculate different metrics that help represent the risk of our
        portfolio and it's performance. The following metrics will be calculated
        in this method:

        1. Standard Deviation of Percent Returns.
        2. Covariance of Percent Returns.
        2. Variance of Percent Returns.
        3. Average Percent Return
        4. Weighted Average Percent Return.
        5. Portfolio Variance.

        Returns:
        ----
        dict -- [description]
        """

        if not self._stock_frame_daily:
            self._grab_daily_historical_prices()

        # Calculate the weights.
        porftolio_weights = self.portfolio_weights()

        # Calculate the Daily Returns (%)
        self._stock_frame_daily.frame['daily_returns_pct'] = self._stock_frame_daily.symbol_groups['close'].transform(
            lambda x: x.pct_change()
        )

        # Calculate the Daily Returns (Mean)
        self._stock_frame_daily.frame['daily_returns_avg'] = self._stock_frame_daily.symbol_groups['daily_returns_pct'].transform(
            lambda x: x.mean()
        )

        # Calculate the Daily Returns (Standard Deviation)
        self._stock_frame_daily.frame['daily_returns_std'] = self._stock_frame_daily.symbol_groups['daily_returns_pct'].transform(
            lambda x: x.std()
        )

        # Calculate the Covariance.
        returns_cov = self._stock_frame_daily.frame.unstack(
            level=0)['daily_returns_pct'].cov()

        # Take the other columns and get ready to add them to our dictionary.
        returns_avg = self._stock_frame_daily.symbol_groups['daily_returns_avg'].tail(
            n=1
        ).to_dict()

        returns_std = self._stock_frame_daily.symbol_groups['daily_returns_std'].tail(
            n=1
        ).to_dict()

        metrics_dict = {}

        portfolio_variance = self.portfolio_variance(
            weights=porftolio_weights,
            covariance_matrix=returns_cov
        )

        for index_tuple in returns_std:

            symbol = index_tuple[0]
            metrics_dict[symbol] = {}
            metrics_dict[symbol]['weight'] = porftolio_weights[symbol]
            metrics_dict[symbol]['average_returns'] = returns_avg[index_tuple]
            metrics_dict[symbol]['weighted_returns'] = returns_avg[index_tuple] * \
                metrics_dict[symbol]['weight']
            metrics_dict[symbol]['standard_deviation_of_returns'] = returns_std[index_tuple]
            metrics_dict[symbol]['variance_of_returns'] = returns_std[index_tuple] ** 2
            metrics_dict[symbol]['covariance_of_returns'] = returns_cov.loc[[
                symbol]].to_dict()

        metrics_dict['portfolio'] = {}
        metrics_dict['portfolio']['variance'] = portfolio_variance

        return metrics_dict

    def portfolio_weights(self) -> dict:
        """Calculate the weights for each position in the portfolio

        Returns:
        ----
        {dict} -- Each symbol with their designated weights.
        """

        weights = {}
        symbols = self.positions.keys()
        quotes = self.td_client.get_quotes(instruments=list(symbols))

        # Grab the projected market value.
        projected_market_value_dict = self.projected_market_value(
            current_prices=quotes
        )

        # Calc the weights
        for symbol in projected_market_value_dict:
            if symbol != 'total':
                weights[symbol] = projected_market_value_dict[symbol]['market_value'] / \
                    projected_market_value_dict['total']['market_value']

        return weights

    def portfolio_summary(self):
        """Generates a summary of our portfolio."""

        # First grab all the symbols.
        symbols = self.positions.keys()

        # Grab the quotes.
        quotes = self.td_client.get_quotes(instruments=list(symbols))

        portfolio_summary_dict = {}
        portfolio_summary_dict['projected_market_value'] = self.projected_market_value(
            current_prices=quotes
        )
        
        portfolio_summary_dict['portfolio_weights'] = self.portfolio_weights()
        portfolio_summary_dict['portfolio_risk'] = ""
        return portfolio_summary_dict

    def in_portfolio(self, symbol: str) -> bool:
        """checks if the symbol is in the portfolio.

        Arguments:
        ----
        symbol {str} -- The symbol of the instrument to be deleted. Example: 'AAPL' or '/ES'

        Returns:
        ----
        bool -- `True` if the position is in the portfolio, `False` otherwise.

        Usage:
        ----
            >>> portfolio = Portfolio()
            >>> new_position = Portfolio.add_position(
                symbol='MSFT', 
                asset_type='equity'
            )
            >>> in_position_flag = Portfolio.in_portfolio(symbol='MSFT')
            >>> in_position_flag
                True
        """

        return symbol in self.positions

    def get_ownership_status(self, symbol: str) -> bool:
        """Gets the ownership status for a position in the portfolio.

        Arguments:
        ----
        symbol {str} -- The symbol you want to grab the ownership status for.

        Returns:
        ----
        {bool} -- `True` if the we own the position, `False` if we do not own it.
        """

        if self.in_portfolio(symbol=symbol) and self.positions[symbol]['ownership_status']:
            return self.positions[symbol]['ownership_status']
        else:
            return False

    def set_ownership_status(self, symbol: str, ownership: bool=None) -> None:
        """Sets the ownership status for a position in the portfolio.

        Arguments:
        ----
        symbol {str} -- The symbol you want to change the ownership status for.

        ownership {bool} -- The ownership status you want the symbol to have. Can either
            be `True` or `False`.

        Raises:
        ----
        KeyError: If the symbol does not exist in the portfolio it will return an error.
        """

        try:
            self.positions[symbol]['ownership_status'] = self.in_portfolio(symbol) and \
                                                         self.positions[symbol]['quantity'] != 0
        except KeyError:
            raise KeyError(
                "Can't set ownership status, you either do not have the symbol in your portfolio or no invested shares."
            )

    def is_profitable(self, symbol: str, current_price: float) -> bool:
        """Specifies whether a position is profitable.

        Arguments:
        ----
        symbol {str} -- The symbol of the instrument, to check profitability.

        current_price {float} -- The current trading price of the instrument.

        Returns:
        ----
        {bool} -- Specifies whether the position is profitable or flat `True` or not
            profitable `False`.

        Raises:
        ----
        KeyError: If the Symbol does not exist it will return a key error.

        Usage:
        ----
            >>> portfolio = Portfolio()
            >>> new_position = Portfolio.add_position(
                symbol='MSFT', 
                asset_type='equity',
                purchase_price=4.00,
                purchase_date="2020-01-31"
            )
            >>> is_profitable_flag = Portfolio.is_profitable(
                symbol='MSFT',
                current_price=7.00
            )
            >>> is_profitable_flag
            True
        """

        # Grab the purchase price, if it exists.
        if self.in_portfolio(symbol=symbol):
            purchase_price = self.positions[symbol]['purchase_price']
        else:
            raise KeyError("The Symbol you tried to request does not exist.")

        return purchase_price <= current_price

    def projected_market_value(self, current_prices: dict, date=None) -> dict:
        """Returns the Projected market value for all the positions in the portfolio.

        Arguments:
        ----
        current_prices {dict} -- A dictionary of current quotes for each of the symbols
            in the portfolio.

        Returns:
        ----
        dict -- A summarized version of the portfolio with each position, purchase price, current price,
            and projected values.

        Usage:
        ----
            >>> portfolio = Portfolio()
            >>> new_position = portfolio.add_position(
                symbol='MSFT', 
                asset_type='equity',
                purchase_price=4.00,
                purchase_date="2020-01-31"
            )
            >>> portfolio_summary = portfolio.projected_market_value(current_prices={'MSFT':{'lastPrice': 8.00, 'openPrice': 7.50}})        
        """

        projected_value = {}
        total_value = 0.0
        invested_capital = 0.0
        total_profit_or_loss = 0.0

        position_count_profitable = 0
        position_count_not_profitable = 0
        position_count_break_even = 0

        if not date:
            date = datetime.today()

        for symbol in current_prices:
            if self.in_portfolio(symbol=symbol):
                projected_value[symbol] = {}
                current_quantity = self.positions[symbol]['quantity']
                purchase_price = self.positions[symbol]['purchase_price']
                
                try:
                    current_price = current_prices[symbol]['lastPrice']
                except KeyError:
                    current_price = current_prices[symbol]['close']

                self.positions[symbol]['current_price'] = current_price

                for lot in self.position_lots[symbol]:
                    lot['market_value'] = lot['quantity'] * current_price
                    lot['profit_loss'] = lot['market_value'] - lot['cost_basis']
                    lot['total_return'] = round((lot['profit_loss'] / lot['cost_basis']), 4)
                    lot['annualized_return'] = round((lot['profit_loss'] / lot['cost_basis']) * (365 / (date - datetime.strptime(str(lot['purchase_date']), "%Y-%m-%d %H:%M:%S")).days), 4)
                
                projected_value[symbol]['purchase_price'] = purchase_price
                projected_value[symbol]['current_price'] = current_price
                projected_value[symbol]['quantity'] = current_quantity

                # Calculate total market value.
                projected_value[symbol]['market_value'] = (
                    current_price * current_quantity
                )

                # Calculate total invested capital.
                projected_value[symbol]['invested_capital'] = (
                    current_quantity * purchase_price
                )

                projected_value[symbol]['profit_loss'] = ((current_price - purchase_price) * current_quantity)
                projected_value[symbol]['total_return'] = round(((current_price - purchase_price) / purchase_price), 4)
                projected_value[symbol]['annualized_return'] = round(((current_price - purchase_price) / purchase_price) * (365 / (date - datetime.strptime(str(self.positions[symbol]['purchase_date']), "%Y-%m-%d %H:%M:%S")).days), 4)

                total_value += projected_value[symbol]['market_value']
                total_profit_or_loss += projected_value[symbol]['profit_loss']
                invested_capital += projected_value[symbol]['invested_capital']

                if projected_value[symbol]['profit_loss'] > 0:
                    position_count_profitable += 1
                elif projected_value[symbol]['profit_loss'] < 0:
                    position_count_not_profitable += 1
                else:
                    position_count_break_even += 1

        projected_value['total'] = {}
        projected_value['total']['positions_count'] = len(self.positions)
        projected_value['total']['market_value'] = total_value
        projected_value['total']['invested_capital'] = invested_capital
        projected_value['total']['total_profit_or_loss'] = total_profit_or_loss
        projected_value['total']['total_return'] = round((total_profit_or_loss / invested_capital), 4)
        projected_value['total']['count_profitable_positions'] = position_count_profitable
        projected_value['total']['count_non_profitable_positions'] = position_count_not_profitable
        projected_value['total']['count_breakeven_positions'] = position_count_break_even

        self.market_value = total_value
        self.invested_capital = invested_capital
        self.profit_loss = total_profit_or_loss
        self.total_return = projected_value['total']['total_return']
        return projected_value

    def get_portfolio_max_drawdown(self, window: int=252):
        """Calcs the max drawdown for the given benchmark index.

        window -- the size of the rolling period to use to calc min/max
        drawdown; note there are 252 trading days in a year, so if you want a
        window of 2, 3, 4,... years the window will be 504, 756, 1008,... days
        
        """

        if not self.performance_tracker:
            raise AttributeError('Must create performance tracker before calc max drawdown.')

        total_performance_tracker = self.get_performance_tracker(totals_only=True)
        
        df = DataFrame.from_dict(total_performance_tracker, orient='index')
        
        # calc daily drawdown using the max market val over 'window' days
        rolling_max_mv = df['market_value'].rolling(window=window, min_periods=1).max()
        
        # subtract 1 to make DD negative
        daily_drawdown = df['market_value'] / rolling_max_mv - 1.0

        # then calc max drawdown as min of daily drawdowns over 'window' days
        daily_max_drawdown = daily_drawdown.rolling(window, min_periods=1).min()
        return min(daily_max_drawdown)

    def get_max_drawdown(self, tracker: dict, keys: str, print_stuff=True):
        """My original method fo calculating the max drawdown [DELETE?]"""

        display_min_vals = {}
        display_min_val_dates = {}
        display_max_prior_vals = {}
        display_max_prior_vals_dates = {}
        display_min_portfolio_mv = {}
        display_max_prior_portfolio_mv = {}
        display_max_drawdowns = {}
        
        daily_portfolio_mvs = [(dt, tracker[dt]['total']['market_value']) for dt in tracker if isinstance(tracker[dt], dict)]
        
        for key in keys:
            daily_portfolio_vals = [(dt, tracker[dt]['total'][key]) for dt in tracker if isinstance(tracker[dt], dict)]
            min_portfolio_val = min([x[1] for x in daily_portfolio_vals])
            min_portfolio_val_date = [x[0] for x in daily_portfolio_vals if x[1] == min_portfolio_val][0]
            
            prior_daily_portfolio_vals = [tup for tup in daily_portfolio_vals if tup[0] < min_portfolio_val_date]
            max_prior_portfolio_val = max([tup[1] for tup in prior_daily_portfolio_vals])
            max_prior_portfolio_val_date = [tup[0] for tup in prior_daily_portfolio_vals if tup[1] == max_prior_portfolio_val][0]
            
            min_portfolio_mv = [x[1] for x in daily_portfolio_mvs if x[0] == min_portfolio_val_date][0]
            max_prior_portfolio_mv = [x[1] for x in daily_portfolio_mvs if x[0] == max_prior_portfolio_val_date][0]
            
            max_drawdown = round((min_portfolio_mv - max_prior_portfolio_mv) / max_prior_portfolio_mv, 4)
            
            display_min_vals[f'min_val_(key={key})'] = min_portfolio_val
            display_min_val_dates[f'date_min_val_(key={key})'] = min_portfolio_val_date
            display_max_prior_vals[f'max_prior_val_(key={key})'] = max_prior_portfolio_val
            display_max_prior_vals_dates[f'date_max_prior_val_(key={key})'] = max_prior_portfolio_val_date
            display_min_portfolio_mv[f'min_mv_(key={key})'] = min_portfolio_mv
            display_max_prior_portfolio_mv[f'max_prior_mv_(key={key})'] = max_prior_portfolio_mv
            display_max_drawdowns[f'max_drawdown_(key={key})'] = max_drawdown

        if print_stuff:
            print('--'*50)
            [print(f'{min_val}: {display_min_vals[min_val]}') for min_val in display_min_vals]
            print()
            [print(f'{min_val_date}: {display_min_val_dates[min_val_date]}') for min_val_date in display_min_val_dates]
            print()
            [print(f'{max_val}: {display_max_prior_vals[max_val]}') for max_val in display_max_prior_vals]
            print()
            [print(f'{max_val_date}: {display_max_prior_vals_dates[max_val_date]}') for max_val_date in display_max_prior_vals_dates]
            print()
            [print(f'{min_mv}: {display_min_portfolio_mv[min_mv]}') for min_mv in display_min_portfolio_mv]
            print()
            [print(f'{max_mv}: {display_max_prior_portfolio_mv[max_mv]}') for max_mv in display_max_prior_portfolio_mv]
            print()
            [print(f'{drawdown}: {display_max_drawdowns[drawdown]}') for drawdown in display_max_drawdowns]
            print('--'*50)

        all_drawdowns = [display_max_drawdowns[d] for d in display_max_drawdowns]
        max_drawdown = max(set(all_drawdowns), key = all_drawdowns.count)
        return max_drawdown
    
    @property
    def historical_prices(self) -> List[dict]:
        """Gets the historical prices for the Portfolio

        Returns:
        ----
        List[dict] -- A list of historical candle prices.
        """

        return self._historical_prices

    @historical_prices.setter
    def historical_prices(self, historical_prices: List[dict]) -> None:
        """Sets the historical prices for the Portfolio

        Arguments:
        ----
        historical_prices {List[dict]} -- A list of historical candle prices.
        """

        self._historical_prices = historical_prices

    @property
    def stock_frame(self) -> StockFrame:
        """Gets the StockFrame object for the Portfolio

        Returns:
        ----
        {StockFrame} -- A StockFrame object with symbol groups, and rolling windows.
        """

        return self._stock_frame

    @stock_frame.setter
    def stock_frame(self, stock_frame: StockFrame) -> None:
        """Sets the StockFrame object for the Portfolio

        Arguments:
        ----
        stock_frame {StockFrame} -- A StockFrame object with symbol groups, and rolling windows.
        """

        self._stock_frame = stock_frame

    @property
    def td_client(self) -> TDClient:
        """Gets the TDClient object for the Portfolio

        Returns:
        ----
        {TDClient} -- An authenticated session with the TD API.
        """

        return self._td_client

    @td_client.setter
    def td_client(self, td_client: TDClient) -> None:
        """Sets the TDClient object for the Portfolio

        Arguments:
        ----
        td_client {TDClient} -- An authenticated session with the TD API.
        """

        self._td_client: TDClient = td_client

    def _grab_daily_historical_prices(self) -> StockFrame:
        """Grabs the daily historical prices for each position and sets the daily
        stock frame, which is multi indexed by symbol and datetime

        Returns:
        ----
        {StockFrame} -- A StockFrame object with data organized, grouped, and
        sorted.
        """

        new_prices = []

        # Loop through each position.
        for symbol in self.positions:

            # Grab the historical prices.
            historical_prices_response = self.td_client.get_price_history(
                symbol=symbol,
                period_type='year',
                period=1,
                frequency_type='daily',
                frequency=1,
                extended_hours=True
            )

            # Loop through the chandles.
            for candle in historical_prices_response['candles']:

                new_price_mini_dict = {}
                new_price_mini_dict['symbol'] = symbol
                new_price_mini_dict['open'] = candle['open']
                new_price_mini_dict['close'] = candle['close']
                new_price_mini_dict['high'] = candle['high']
                new_price_mini_dict['low'] = candle['low']
                new_price_mini_dict['volume'] = candle['volume']
                new_price_mini_dict['datetime'] = candle['datetime']
                new_prices.append(new_price_mini_dict)

        # Create and set the StockFrame
        self._stock_frame_daily = StockFrame(data=new_prices)
        self._stock_frame_daily.create_frame()

        return self._stock_frame_daily


# s = 'happy is friday everyone that is awesome how we doing happy how'
# test_set = set([i for i in s.split()])
# print('happy' in test_set)