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

    """For all of the indicators, we use the locals func to get all of the args
    passed into the func, then del self bc we are storing the args for later use
    and the self arg shouldn't be passed into future func calls. These future
    func calls will happen whenever we call the refresh method to update the
    values of the indicators. Finally after storing the func + args we calculate
    the indicator col for each symbol group"""
    
    def change_in_price(self, column_name: str = 'change_in_price') -> pd.DataFrame:
        """Calculates the day-over-day change in close price 

        Returns:
        ----
        {pd.DataFrame} -- A data frame with the Change in Price included.
        """

        locals_data = locals()         # grabs the func args (local vars) that were passed - e.g., self, col_name 
        del locals_data['self']        # del self bc we only care about the other args passed into an indicator func
        
        self._current_indicators[column_name] = {}                             # Add a new indicator to our indicators dict
        self._current_indicators[column_name]['args'] = locals_data            # storing the args to pass to the later func call
        self._current_indicators[column_name]['func'] = self.change_in_price   # storing the func to call it later using 'args'

        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.diff()                                                 # finally, calculate the actual change in close price
        )

        self._frame['log_change_in_price'] = self._price_groups['close'].transform(
            lambda x: np.log(x).diff()
        )
        return self._frame

    def discount_ratio(self, period: int, discounts: list, interval: float = 0.10, column_name: str = 'discount_ratio') -> pd.DataFrame:
        """Calculates the % decline the current price represents from the max
        price over the previous period days

        discounts: list of floats from 0.00 to 1.00, each representing a
        discount threshold from the max period price
        interval: the discount range between which each indicator will produce a
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
        
        # drop rows that won't have indicator signals
        idx_names = self._frame[(self._frame['discount_ratio'] > max(discounts)) & (self._frame['discount_ratio'] < 1)].index
        self._frame.drop(idx_names, inplace=True)

        for discount in discounts:
            col = f'discount_ratio_{str(int(discount*100))}'
            self._frame[col] = np.where((self._frame['discount_ratio'] <= discount), 1, -1)
            # self._frame[col] = np.where((self._frame['discount_ratio'] <= discount) & \
            #                             (self._frame['discount_ratio'] > (discount - interval)), 1, -1)

        self._frame.dropna(inplace=True)
        return self._frame

    # def first_discount_return(self, required_return: float, discounts: list, column_name='first_discount_return'):
    #     """Calcs the % return on the first (highest) discount ratio"""
        
    #     locals_data = locals()
    #     del locals_data['self']

    #     self._current_indicators[column_name] = {}
    #     self._current_indicators[column_name]['args'] = locals_data
    #     self._current_indicators[column_name]['func'] = self.first_discount_return
        
    #     first_discount = max(discounts)
    #     first_discount_col = f'discount_ratio_{str(int(first_discount*100))}'
    #     self._frame['dates_discount_reached'] = np.where((self._frame[first_discount_col] == 1), self._frame.index[1], 0)
        
    #     self._frame[column_name] = self._price_groups['close'].transform(
    #         lambda x: x.pct_change(periods=period)
    #     )
        
        
    #     col = f'return_on_discount_{str(int(first_discount*100))}'
    #     self._frame[f'cost_basis_{str(int(first_discount*100))}'] = 
    #     self._frame[col] = 
    
    def close_to_avg_ratio(self, period: int, premium: float = 1.20, column_name: str = 'close_to_avg_ratio') -> pd.DataFrame:
        """Calculates the ratio of current price to the avg close price over the
        period

        period: the num of days to track premium: the pct above period avg when
        to buy

        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.close_to_avg_ratio
        
        self._frame['period_avg'] = self._price_groups['close'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        self._frame['close_to_avg_ratio'] = self._frame['close'] / self._frame['period_avg']
        col = f'close_to_avg_{str(int(premium*100))}'
        self._frame[col] = np.where(self._frame['close_to_avg_ratio'] >= premium, 1, -1)
        self._frame.dropna(inplace=True)
        return self._frame

    
    def rsi(self, period: int, method: str = 'wilders', column_name: str = 'rsi') -> pd.DataFrame:
        """Calculates the Relative Strength Index (RSI).

        Arguments:
        ----
        period {int} -- The number of periods to use to calculate the RSI.

        Keyword Arguments:
        ----
        method {str} -- The calculation methodology. (default: {'wilders'})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the RSI indicator included.

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
            >>> indicator_client.rsi(period=14)
            >>> price_data_frame = inidcator_client.price_data_frame
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.rsi

        # First calculate the Change in Price.
        if 'change_in_price' not in self._frame.columns:
            self.change_in_price()

        # Define the up days.
        self._frame['up_day'] = self._price_groups['change_in_price'].transform(
            lambda x : np.where(x >= 0, x, 0)
        )

        # Define the down days.
        self._frame['down_day'] = self._price_groups['change_in_price'].transform(
            lambda x : np.where(x < 0, x.abs(), 0)
        )

        # Calculate the EWMA for the Up days.
        self._frame['ewma_up'] = self._price_groups['up_day'].transform(
            lambda x: x.ewm(span = period).mean()
        )

        # Calculate the EWMA for the Down days.
        self._frame['ewma_down'] = self._price_groups['down_day'].transform(
            lambda x: x.ewm(span = period).mean()
        )

        # Calculate the Relative Strength
        relative_strength = self._frame['ewma_up'] / self._frame['ewma_down']

        # Calculate the Relative Strength Index
        relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

        # Add the info to the data frame.
        self._frame['rsi'] = np.where(relative_strength_index == 0, 100, 100 - (100 / (1 + relative_strength_index)))

        # Clean up before sending back.
        self._frame.drop(
            labels=['ewma_up', 'ewma_down', 'down_day', 'up_day', 'change_in_price'],
            axis=1,
            inplace=True
        )

        return self._frame

    def sma(self, period: int, column_name: str = 'sma') -> pd.DataFrame:
        """Calculates the Simple Moving Average (SMA).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating the SMA.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the SMA indicator included.

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
            >>> indicator_client.sma(period=100)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.sma

        # Add the SMA
        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        return self._frame

    def ema(self, period: int, alpha: float = 0.0, column_name = 'ema') -> pd.DataFrame:
        """Calculates the Exponential Moving Average (EMA).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating the EMA.

        alpha {float} -- The alpha weight used in the calculation. (default: {0.0})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the EMA indicator included.

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
            >>> indicator_client.ema(period=50, alpha=1/50)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.ema

        # Add the EMA
        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.ewm(span=period).mean()
        )

        return self._frame

    def rate_of_change(self, period: int = 1, column_name: str = 'rate_of_change') -> pd.DataFrame:
        """Calculates the Rate of Change (ROC).

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
            >>> indicator_client.rate_of_change()
        """
        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.rate_of_change

        # Add the Momentum indicator.
        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.pct_change(periods=period)
        )

        return self._frame        

    def bollinger_bands(self, period: int = 20, column_name: str = 'bollinger_bands') -> pd.DataFrame:
        """Calculates the Bollinger Bands.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating 
            the Bollinger Bands. (default: {20})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Lower and Upper band
            indicator included.

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
            >>> indicator_client.bollinger_bands()
        """
        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.bollinger_bands

        # Define the Moving Avg.
        self._frame['moving_avg'] = self._price_groups['close'].transform(
            lambda x : x.rolling(window=period).mean()
        )

        # Define Moving Std.
        self._frame['moving_std'] = self._price_groups['close'].transform(
            lambda x : x.rolling(window=period).std()
        )

        # Define the Upper Band.
        self._frame['band_upper'] = 4 * (self._frame['moving_std'] / self._frame['moving_avg'])

        # Define the lower band
        self._frame['band_lower'] = (
            (self._frame['close'] - self._frame['moving_avg']) + 
            (2 * self._frame['moving_std']) / 
            (4 * self._frame['moving_std'])
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['moving_avg', 'moving_std'],
            axis=1,
            inplace=True
        )

        return self._frame   

    def average_true_range(self, period: int = 14, column_name: str ='average_true_range') -> pd.DataFrame:
        """Calculates the Average True Range (ATR).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating 
            the ATR. (default: {14})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the ATR included.

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
            >>> indicator_client.average_true_range()
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.average_true_range


        # Calculate the different parts of True Range.
        self._frame['true_range_0'] = abs(self._frame['high'] - self._frame['low'])
        self._frame['true_range_1'] = abs(self._frame['high'] - self._frame['close'].shift())
        self._frame['true_range_2'] = abs(self._frame['low'] - self._frame['close'].shift())

        # Grab the Max.
        self._frame['true_range'] = self._frame[['true_range_0', 'true_range_1', 'true_range_2']].max(axis=1)

        # Calculate the Average True Range.
        self._frame['average_true_range'] = self._frame['true_range'].transform(
            lambda x: x.ewm(span = period, min_periods = period).mean()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['true_range_0', 'true_range_1', 'true_range_2', 'true_range'],
            axis=1,
            inplace=True
        )

        return self._frame

    def stochastic_oscillator(self, column_name: str = 'stochastic_oscillator') -> pd.DataFrame:
        """Calculates the Stochastic Oscillator.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Stochastic Oscillator included.

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
            >>> indicator_client.stochastic_oscillator()
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.stochastic_oscillator

        # Calculate the stochastic_oscillator.
        self._frame['stochastic_oscillator'] = (
            self._frame['close'] - self._frame['low'] / 
            self._frame['high'] - self._frame['low']
        )

        return self._frame 

    def macd(self, fast_period: int = 12, slow_period: int = 26, column_name: str = 'macd') -> pd.DataFrame:
        """Calculates the Moving Average Convergence Divergence (MACD).

        Arguments:
        ----
        fast_period {int} -- The number of periods to use when calculating 
            the fast moving MACD. (default: {12})

        slow_period {int} -- The number of periods to use when calculating 
            the slow moving MACD. (default: {26})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the MACD included.

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
            >>> indicator_client.macd(fast_period=12, slow_period=26)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.macd

        # Calculate the Fast Moving MACD.
        self._frame['macd_fast'] = self._frame['close'].transform(
            lambda x: x.ewm(span = fast_period, min_periods = fast_period).mean()
        )

        # Calculate the Slow Moving MACD.
        self._frame['macd_slow'] = self._frame['close'].transform(
            lambda x: x.ewm(span = slow_period, min_periods = slow_period).mean()
        )

        # Calculate the difference between the fast and the slow.
        self._frame['macd_diff'] = self._frame['macd_fast'] - self._frame['macd_slow']

        # Calculate the Exponential moving average of the fast.
        self._frame['macd'] = self._frame['macd_diff'].transform(
            lambda x: x.ewm(span = 9, min_periods = 8).mean()
        )

        return self._frame 

    def mass_index(self, period: int = 9, column_name: str = 'mass_index') -> pd.DataFrame:
        """Calculates the Mass Index indicator.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating 
            the mass index. (default: {9})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Mass Index included.

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
            >>> indicator_client.mass_index(period=9)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.mass_index

        # Calculate the Diff.
        self._frame['diff'] = self._frame['high'] - self._frame['low']

        # Calculate Mass Index 1
        self._frame['mass_index_1'] = self._frame['diff'].transform(
            lambda x: x.ewm(span = period, min_periods = period - 1).mean()
        )

        # Calculate Mass Index 2
        self._frame['mass_index_2'] = self._frame['mass_index_1'].transform(
            lambda x: x.ewm(span = period, min_periods = period - 1).mean()
        )
        
        # Grab the raw index.
        self._frame['mass_index_raw'] = self._frame['mass_index_1'] / self._frame['mass_index_2']

        # Calculate the Mass Index.
        self._frame['mass_index'] = self._frame['mass_index_raw'].transform(
            lambda x: x.rolling(window=25).sum()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['diff', 'mass_index_1', 'mass_index_2', 'mass_index_raw'],
            axis=1,
            inplace=True
        )

        return self._frame
    
    def force_index(self, period: int, column_name: str = 'force_index') -> pd.DataFrame:
        """Calculates the Force Index.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating 
            the force index.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the force index included.

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
            >>> indicator_client.force_index(period=9)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.force_index

        # Calculate the Force Index.
        self._frame[column_name] = self._frame['close'].diff(period)  * self._frame['volume'].diff(period)

        return self._frame

    def ease_of_movement(self, period: int, column_name: str = 'ease_of_movement') -> pd.DataFrame:
        """Calculates the Ease of Movement.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating 
            the Ease of Movement.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Ease of Movement included.

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
            >>> indicator_client.ease_of_movement(period=9)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.ease_of_movement
        
        # Calculate the ease of movement.
        high_plus_low = (self._frame['high'].diff(1) + self._frame['low'].diff(1))
        diff_divi_vol = (self._frame['high'] - self._frame['low']) / (2 * self._frame['volume'])
        self._frame['ease_of_movement_raw'] = high_plus_low * diff_divi_vol

        # Calculate the Rolling Average of the Ease of Movement.
        self._frame['ease_of_movement'] = self._frame['ease_of_movement_raw'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['ease_of_movement_raw'],
            axis=1,
            inplace=True
        )

        return self._frame

    def commodity_channel_index(self, period: int, column_name: str = 'commodity_channel_index') -> pd.DataFrame:
        """Calculates the Commodity Channel Index.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating 
            the Commodity Channel Index.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Commodity Channel Index included.

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
            >>> indicator_client.commodity_channel_index(period=9)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.commodity_channel_index

        # Calculate the Typical Price.
        self._frame['typical_price'] = (self._frame['high'] + self._frame['low'] + self._frame['close']) / 3

        # Calculate the Rolling Average of the Typical Price.
        self._frame['typical_price_mean'] = self._frame['pp'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # Calculate the Rolling Standard Deviation of the Typical Price.
        self._frame['typical_price_std'] = self._frame['pp'].transform(
            lambda x: x.rolling(window=period).std()
        )

        # Calculate the Commodity Channel Index.
        self._frame[column_name] = self._frame['typical_price_mean'] / self._frame['typical_price_std']

        # Clean up before sending back.
        self._frame.drop(
            labels=['typical_price', 'typical_price_mean', 'typical_price_std'],
            axis=1,
            inplace=True
        )

        return self._frame

    def standard_deviation(self, period: int, column_name: str = 'standard_deviation') -> pd.DataFrame:
        """Calculates the Standard Deviation.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating 
            the standard deviation.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Standard Deviation included.

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
            >>> indicator_client.standard_deviation(period=9)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.standard_deviation

        # Calculate the Standard Deviation.
        self._frame[column_name] = self._frame['close'].transform(
            lambda x: x.ewm(span=period).std()
        )

        return self._frame

    def chaikin_oscillator(self, period: int, column_name: str = 'chaikin_oscillator') -> pd.DataFrame:
        """Calculates the Chaikin Oscillator.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating 
            the Chaikin Oscillator.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Chaikin Oscillator included.

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
            >>> indicator_client.chaikin_oscillator(period=9)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.chaikin_oscillator

        # Calculate the Money Flow Multiplier.
        money_flow_multiplier_top = 2 * (self._frame['close'] - self._frame['high'] - self._frame['low'])
        money_flow_multiplier_bot = (self._frame['high'] - self._frame['low'])

        # Calculate Money Flow Volume
        self._frame['money_flow_volume'] = (money_flow_multiplier_top / money_flow_multiplier_bot) * self._frame['volume']

        # Calculate the 3-Day moving average of the Money Flow Volume.
        self._frame['money_flow_volume_3'] = self._frame['money_flow_volume'].transform(
            lambda x: x.ewm(span=3, min_periods=2).mean()
        )

        # Calculate the 10-Day moving average of the Money Flow Volume.
        self._frame['money_flow_volume_10'] = self._frame['money_flow_volume'].transform(
            lambda x: x.ewm(span=10, min_periods=9).mean()
        )

        # Calculate the Chaikin Oscillator.
        self._frame[column_name] = self._frame['money_flow_volume_3'] - self._frame['money_flow_volume_10']

        # Clean up before sending back.
        self._frame.drop(
            labels=['money_flow_volume_3', 'money_flow_volume_10', 'money_flow_volume'],
            axis=1,
            inplace=True
        )

        return self._frame

    def kst_oscillator(self, r1: int, r2: int, r3: int, r4: int, n1: int, n2: int, n3: int, n4: int, column_name: str = 'kst_oscillator') -> pd.DataFrame:
        """Calculates the Mass Index indicator.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating 
            the mass index. (default: {9})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Mass Index included.

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
            >>> indicator_client.mass_index(period=9)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.kst_oscillator

        # Calculate the ROC 1.
        self._frame['roc_1'] = self._frame['close'].diff(r1 - 1)  / self._frame['close'].shift(r1 - 1)

        # Calculate the ROC 2.
        self._frame['roc_2'] = self._frame['close'].diff(r2 - 1)  / self._frame['close'].shift(r2 - 1)

        # Calculate the ROC 3.
        self._frame['roc_3'] = self._frame['close'].diff(r3 - 1)  / self._frame['close'].shift(r3 - 1)

        # Calculate the ROC 4.
        self._frame['roc_4'] = self._frame['close'].diff(r4 - 1)  / self._frame['close'].shift(r4 - 1)


        # Calculate the Mass Index.
        self._frame['roc_1_n'] = self._frame['roc_1'].transform(
            lambda x: x.rolling(window=n1).sum()
        )

        # Calculate the Mass Index.
        self._frame['roc_2_n'] = self._frame['roc_2'].transform(
            lambda x: x.rolling(window=n2).sum()
        )

        # Calculate the Mass Index.
        self._frame['roc_3_n'] = self._frame['roc_3'].transform(
            lambda x: x.rolling(window=n3).sum()
        )

        # Calculate the Mass Index.
        self._frame['roc_4_n'] = self._frame['roc_4'].transform(
            lambda x: x.rolling(window=n4).sum()
        )

        self._frame[column_name] = 100 * (self._frame['roc_1_n'] + 2 * self._frame['roc_2_n'] + 3 * self._frame['roc_3_n'] + 4 * self._frame['roc_4_n'])
        self._frame[column_name + "_signal"] = self._frame['column_name'].transform(
            lambda x: x.rolling().mean()
        )
        
        # Clean up before sending back.
        self._frame.drop(
            labels=['roc_1', 'roc_2', 'roc_3', 'roc_4', 'roc_1_n', 'roc_2_n', 'roc_3_n', 'roc_4_n'],
            axis=1,
            inplace=True
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

    def check_signals(self) -> Union[pd.DataFrame, None]:
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

