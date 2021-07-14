import numpy as np
import pandas as pd

from typing import Any
from typing import Dict
from typing import Union

from pyrobot.stock_frame import StockFrame

class Indicators():

    """
    Represents an Indicator Object which can be used to easily add technical
    indicators to a StockFrame. (entirely reliant on StockFrame obj; w/o the
    StockFrame there can be not indicators)
    """    
    
    def __init__(self, price_data_frame: StockFrame) -> None:
        """Initalizes the Indicator Client.

        Arguments:
        ----
        price_data_frame {pyrobot.StockFrame} -- The price data frame which is used to add indicators to.
            At a minimum this data frame must have the following columns: `['timestamp','close','open','high','low']`.
        
        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.price_data_frame
        """

        self._stock_frame: StockFrame = price_data_frame             # StockFrame obj
        self._price_groups = self._stock_frame.symbol_groups         # DataFrame Groupby obj, grouped by symbol
        self._current_indicators = {}                                # Stores the args and the func obj for each indicator; these are called from refresh method
        self._indicator_signals = {}                                 # Stores the buy/sell signal thresholds for each indicator
        self._frame = self._stock_frame.frame                        # DataFrame obj

        self._indicators_comp_key = []
        self._indicators_key = []
        
        if self.is_multi_index:
            True

    def get_indicator_signal(self, indicator: str= None) -> Dict:
        """Return the raw Pandas Dataframe Object.

        Arguments:
        ----
        indicator {Optional[str]} -- The indicator key, for example `ema` or `sma`.

        Returns:
        ----
        {dict} -- Either all of the indicators or the specified indicator.
        """

        if indicator and indicator in self._indicator_signals:
            return self._indicator_signals[indicator]
        else:
            return self._indicator_signals
    
    def set_indicator_signal(self, indicator: str, buy: float, sell: float, condition_buy: Any, condition_sell: Any, buy_max: float = None, 
                             sell_max: float = None, condition_buy_max: Any = None, condition_sell_max: Any = None) -> None:
        """Used to set an indicator where one indicator crosses above or below a certain numerical threshold.

        Arguments:
        ----
        indicator {str} -- The indicator key, for example `ema` or `sma`.

        buy {float} -- The buy signal threshold for the indicator.
        
        sell {float} -- The sell signal threshold for the indicator.

        condition_buy {str} -- The operator which is used to evaluate the `buy` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.
        
        condition_sell {str} -- The operator which is used to evaluate the `sell` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.

        buy_max {float} -- If the buy threshold has a maximum value that needs to be set, then set the `buy_max` threshold.
            This means if the signal exceeds this amount it WILL NOT PURCHASE THE INSTRUMENT. (defaults to None).
        
        sell_max {float} -- If the sell threshold has a maximum value that needs to be set, then set the `buy_max` threshold.
            This means if the signal exceeds this amount it WILL NOT SELL THE INSTRUMENT. (defaults to None).

        condition_buy_max {str} -- The operator which is used to evaluate the `buy_max` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`. (defaults to None).
        
        condition_sell_max {str} -- The operator which is used to evaluate the `sell_max` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`. (defaults to None).
        """

        # Add the key if it doesn't exist.
        if indicator not in self._indicator_signals:
            self._indicator_signals[indicator] = {}
            self._indicators_key.append(indicator)

        # Add the signals.
        self._indicator_signals[indicator]['buy'] = buy     
        self._indicator_signals[indicator]['sell'] = sell
        self._indicator_signals[indicator]['buy_operator'] = condition_buy
        self._indicator_signals[indicator]['sell_operator'] = condition_sell

        # Add the max signals
        self._indicator_signals[indicator]['buy_max'] = buy_max  
        self._indicator_signals[indicator]['sell_max'] = sell_max
        self._indicator_signals[indicator]['buy_operator_max'] = condition_buy_max
        self._indicator_signals[indicator]['sell_operator_max'] = condition_sell_max
    
    def set_indicator_signal_compare(self, indicator_1: str, indicator_2: str, condition_buy: Any, condition_sell: Any) -> None:
        """Used to set an indicator where one indicator is compared to another indicator.

        Overview:
        ----
        Some trading strategies depend on comparing one indicator to another indicator.
        For example, the Simple Moving Average crossing above or below the Exponential
        Moving Average. This will be used to help build those strategies that depend
        on this type of structure.

        Arguments:
        ----
        indicator_1 {str} -- The first indicator key, for example `ema` or `sma`.

        indicator_2 {str} -- The second indicator key, this is the indicator we will compare to. For example,
            is the `sma` greater than the `ema`.

        condition_buy {str} -- The operator which is used to evaluate the `buy` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.
        
        condition_sell {str} -- The operator which is used to evaluate the `sell` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.
        """

        # Define the key.
        key = "{ind_1}_comp_{ind_2}".format(
            ind_1=indicator_1,
            ind_2=indicator_2
        )

        # Add the key if it doesn't exist.
        if key not in self._indicator_signals:
            self._indicator_signals[key] = {}
            self._indicators_comp_key.append(key)   

        # Grab the dictionary.
        indicator_dict = self._indicator_signals[key]

        # Add the signals.
        indicator_dict['type'] = 'comparison'
        indicator_dict['indicator_1'] = indicator_1
        indicator_dict['indicator_2'] = indicator_2
        indicator_dict['buy_operator'] = condition_buy
        indicator_dict['sell_operator'] = condition_sell

    @property
    def price_data_frame(self) -> pd.DataFrame:
        """Return the raw Pandas Dataframe Object.

        Returns:
        ----
        {pd.DataFrame} -- A multi-index data frame.
        """

        return self._frame

    @price_data_frame.setter
    def price_data_frame(self, price_data_frame: pd.DataFrame) -> None:
        """Sets the price data frame.

        Arguments:
        ----
        price_data_frame {pd.DataFrame} -- A multi-index data frame.
        """

        self._frame = price_data_frame

    @property
    def is_multi_index(self) -> bool:
        """Specifies whether the data frame is a multi-index dataframe.

        Returns:
        ----
        {bool} -- `True` if the data frame is a `pd.MultiIndex` object. `False` otherwise.
        """

        if isinstance(self._frame.index, pd.MultiIndex):
            return True
        else:
            return False
    
    def price_delta(self, period: int = 1, column_name: str = 'price_delta') -> pd.DataFrame:
        """Calculates the $ change in close price over 'period' days

        Returns:
        ----
        {pd.DataFrame} -- A data frame with the Change in Price included.
        """

        locals_data = locals()         # grabs the func args (local vars) that were passed - e.g., self, col_name 
        del locals_data['self']        # del self bc we only care about the other args passed into an indicator func
        
        self._current_indicators[column_name] = {}                             # Add a new indicator to our indicators dict
        self._current_indicators[column_name]['args'] = locals_data            # storing the args to pass to the later func call
        self._current_indicators[column_name]['func'] = self.price_delta   # storing the func to call it later using 'args'

        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.diff(periods=period)                                                 # finally, calculate the actual change in close price
        )

        self._frame['log_price_delta'] = self._price_groups['close'].transform(
            lambda x: np.log(x).diff(periods=period)
        )
        return self._frame

    def discount_ratio(self, period: int, discounts: list, interval: float = 0.10, column_name: str = 'discount_ratio') -> pd.DataFrame:
        """Calcs the % discount the current price represents from the max
        close price over the previous period days

        discounts -- list of floats from 0.00 to 1.00, each representing a
        discount threshold from the max period price
        interval -- the discount range between which each indicator will produce a
        signal
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.discount_ratio
        
        self._frame['period_max'] = self._price_groups['close'].transform(
            lambda x: x.rolling(window=period).max()
        )

        self._frame['discount_ratio'] = 1 - abs((self._frame['close'] - self._frame['period_max'])) / self._frame['period_max']
        
        # # drop rows that won't have signals (can't drop rows bc skews max DD)
        # idx_names = self._frame[(self._frame['discount_ratio'] > max(discounts)) & (self._frame['close_to_sma'] < 1)].index
        # self._frame.drop(idx_names, inplace=True)

        # add a col for each of the discount ratios
        for discount in discounts:
            col = f'max_disc_{str(int(discount*100))}'
            self._frame[col] = np.where((self._frame['discount_ratio'] <= discount), 1, -1)

        return self._frame
    
    def sma_premium(self, period: int, premium: float, column_name: str = 'sma_premium') -> pd.DataFrame:
        """Calculates the ratio of current price to the avg close price over the
        period

        period: the num of days to track premium: the pct above period avg when
        to buy

        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.sma_premium
        
        self._frame['sma'] = self._price_groups['close'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        self._frame['close_to_sma'] = self._frame['close'] / self._frame['sma']
        col = f'sma_prem_{str(int(premium*100))}'
        self._frame[col] = np.where(self._frame['close_to_sma'] >= premium, 1, -1)
        
        # self._frame.dropna(inplace=True)
        return self._frame

    def pct_change(self, period: int = 1, column_name: str = 'pct_change') -> pd.DataFrame:
        """Calc the % change in the close price over 'period' days

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating 
            the ROC. (default: {1})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the ROC indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.pct_change()
        """
        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.pct_change

        # Add the Momentum indicator.
        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.pct_change(periods=period)
        )

        return self._frame        

    def refresh(self):
        """Updates the Indicator columns after adding the new rows."""

        # First update the groups since, we have new rows.
        self._price_groups = self._stock_frame.symbol_groups

        # Grab all the details of the indicators so far.
        for indicator in self._current_indicators:
            
            # Grab the arguments.
            indicator_argument = self._current_indicators[indicator]['args']

            # Grab the function.
            indicator_function = self._current_indicators[indicator]['func']

            # Update the function.
            indicator_function(**indicator_argument)

    def check_signals(self, locked_indicators = None) -> Union[pd.DataFrame, None]:
        """Checks to see if any signals have been generated.

        Returns:
        ----
        {Union[pd.DataFrame, None]} -- If signals are generated then a pandas.DataFrame
            is returned otherwise nothing is returned.
        """

        signals_df = self._stock_frame._check_signals(
            indicators=self._indicator_signals,
            indicators_comp_key=self._indicators_comp_key,
            indicators_key=self._indicators_key
        )

        return signals_df

    def check_current_signals(self, bar, locked_indicators: set = None) -> Union[pd.DataFrame, None]:
        """Checks to see if any signals have been generated.

        Returns:
        ----
        {Union[pd.DataFrame, None]} -- If signals are generated then a pandas.DataFrame
            is returned otherwise nothing is returned.
        """

        signals_df = self._stock_frame._check_current_signals(
            bar=bar,
            indicators=self._indicator_signals,
            indicators_comp_key=self._indicators_comp_key,
            indicators_key=self._indicators_key,
            locked_indicators=locked_indicators
        )

        return signals_df

