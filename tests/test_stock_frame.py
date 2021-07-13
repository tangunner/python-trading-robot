# """Unit test module for the StockFrame Object.

# Will perform an instance test to make sure it creates it. Additionally,
# it will test different properties and methods of the object.
# """

# import unittest
# import pandas as pd
# from unittest import TestCase
# from datetime import datetime
# from datetime import timedelta
# from configparser import ConfigParser

# from pyrobot.robot import PyRobot
# from pyrobot.stock_frame import StockFrame


# class PyRobotStockFrameTest(TestCase):

#     """Will perform a unit test for the StockFrame Object."""

#     def setUp(self) -> None:
#         """Set up the Stock Frame."""

#         # Grab configuration values.
#         config = ConfigParser()
#         config.read('configs/config.ini')       

#         CLIENT_ID = config.get('main', 'CLIENT_ID')
#         REDIRECT_URI = config.get('main', 'REDIRECT_URI')
#         CREDENTIALS_PATH = config.get('main', 'JSON_PATH')

#         # Create a robot.
#         self.robot = PyRobot(
#             client_id = CLIENT_ID, 
#             redirect_uri = REDIRECT_URI, 
#             credentials_path = CREDENTIALS_PATH
#         )

#         # Grab historical prices, first define the start date and end date.
#         start_date = datetime.today()
#         end_date = start_date - timedelta(days=30)

#         # Grab the historical prices.
#         historical_prices = self.robot.grab_historical_prices(
#             start=end_date,
#             end=start_date,
#             bar_size=1,
#             bar_type='minute',
#             symbols=['AAPL','MSFT']
#         )

#         # Convert data to a Data Frame.
#         self.stock_frame = self.robot.create_stock_frame(data=historical_prices['aggregated'])

#     def test_creates_instance_of_session(self):
#         """Create an instance and make sure it's a StockFrame."""

#         self.assertIsInstance(self.stock_frame, StockFrame)

#     def test_frame_property(self):
#         """Test that the `frame` property returns a Pandas DataFrame object."""

#         self.assertIsInstance(self.stock_frame.frame, pd.DataFrame)
#         self.assertIsInstance(self.stock_frame.frame.index, pd.MultiIndex)

#     def test_frame_symbols(self):
#         """Test that the `frame.index` property contains the specified symbols."""

#         self.assertIn('AAPL', self.stock_frame.frame.index)
#         self.assertIn('MSFT', self.stock_frame.frame.index)

#     def test_symbol_groups_property(self):
#         """Test that the `symbol_groups` property returns a Pandas DataFrameGroupBy object."""

#         self.assertIsInstance(self.stock_frame.symbol_groups, pd.core.groupby.DataFrameGroupBy)
    
#     def test_symbol_rolling_groups_property(self):
#         """Test that the `symbol_rolling_groups` property returns a Pandas RollingGroupBy object."""

#         self.assertIsInstance(self.stock_frame.symbol_rolling_groups(size=15), pd.core.window.RollingGroupby)

#     def test_add_row(self):
#         """Test adding a new row to our data frame."""

#         # Define a new row.
#         new_row_dict = {
#             'AAPL':{
#                 'openPrice':100.00,
#                 'closePrice':100.00,
#                 'highPrice':100.00,
#                 'lowPrice':100.00,
#                 'askSize':100,
#                 'bidSize':100,
#                 'quoteTimeInLong':1586390399572
#             }

#         }

#         # Add the row.
#         self.stock_frame.add_rows(data=new_row_dict)

#         # Create a timestamp.
#         time_stamp_parsed = pd.to_datetime(1586390399572, unit='ms', origin='unix')
#         index_tuple = ('AAPL', time_stamp_parsed)

#         # Check to see if the Tuple is in the Index.
#         self.assertIn(index_tuple, self.stock_frame.frame.index)

#     def tearDown(self) -> None:
#         """Teardown the StockFrame."""

#         self.stock_frame = None


# if __name__ == '__main__':
#     unittest.main()




import pandas as pd
import numpy as np

# df = pd.DataFrame(
#     {
#         "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
#         "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
#         "C": np.random.randn(8),
#         "D": np.random.randn(8)
#     }
# )

# grouped = df.groupby(["B"]).sum()
# grouped = df.groupby(["B"]).get_group("one")

# df.groupby("B").groups

# grouped = df.groupby(["A", "B"])
# print(grouped.groups)


# def get_letter_type(letter):
#     if letter.lower() in 'aeiou':
#         return 'vowel'
#     else:
#         return 'consonant'

# grouped = df.groupby(get_letter_type, axis=1)
# grouped.groups


#------------------------------------------
######### GroupBy with MultiIndex #########

# arrays = [
#     ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
#     ["one", "two", "one", "two", "one", "two", "one", "two"],
# ]

# index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
# s = pd.Series(np.random.randn(8), index=index)
# # s

# grouped = s.groupby(level=0)
# grouped.sum()

# df = pd.DataFrame({"A": [1, 1, 1, 1, 2, 2, 3, 3], "B": np.arange(8)}, index=index)
# df

"""NOTE: dfs can be grouped by specifying col names as strings and index levels
as Grouper objs"""

# df.groupby([pd.Grouper(level=1), "A"]).sum()
# # groupby sets both the level 1 idx ("second") and col "A" as the level 1 and 2
# # idx, respectively. Then all of the corresponding values in col "B" that
# # overlap idx vals are summed together

# df.groupby(["second", "A"]).sum()
# # idx labels can instead just be specified as keys to groupby instead of
# # creating a Grouper obj



#-------------------------------------------
######### Iterating Through Groups #########


# df = pd.DataFrame(
#     {
#         "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
#         "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
#         "C": np.random.randn(8),
#         "D": np.random.randn(8)
#     }
# )

# # iterating through a grouping that has one key
# grouped = df.groupby('A')

# for name, group in grouped:
#     print(name)
#     print(group)


# # iterating through a grouping that has multiple keys
# for name, group in df.groupby(['A', 'B']):
#     print(name)
#     print(group)     # the group name is a tuple when there's mult keys

from datetime import datetime

df = pd.DataFrame(
    {
        "symbol": ["SPY", "IBM", "SPY", "SPY", "IBM", "SPY", "AAPL", "AAPL"],
        "close_price": [55.1, 48.3, 17.0, 16.4, 4.96, 8.90, 3.33, 47.78],
        "datetime": [datetime(2003,5,13), datetime(2005,3,22),datetime(1995,8,13),datetime(2011,7,13),datetime(2008,5,8),datetime(2015,11,13),datetime(2021,4,3),datetime(2012,5,5)],
        "stochastics_lol": np.random.randn(8)
    }
)

price_df = df.set_index(keys=['symbol', 'datetime'])

# print(price_df)

print(price_df.tail(50))

# symbol_groups = price_df.groupby(
#     by='datetime',
#     as_index=False,
#     sort=True
# )

symbol_groups = price_df.groupby(
    by='symbol',
    as_index=False,
    sort=True
)

print(len(symbol_groups.tail(1)))
print(symbol_groups.tail(1))
# print(symbol_groups.head(50))

min_prices = symbol_groups.apply(lambda x: x['close_price'].idxmin())
print(min_prices)
# max_prices = symbol_groups.apply(lambda x: x['close_price'][x['close_price'].idxmax()])

def cum_ret(x, y):
    return x * (1 + y)



# for name, group in symbol_groups:
#     print(name)
#     print(group)

# for name, group in df.groupby(['symbol', 'datetime']):
#     print(name)
#     print(group)

#------------------------------
######### Aggregation #########

# grouped = df.groupby(["A", "B"], as_index=False)
# grouped.size()

# grouped.describe()



import random

tmp = pd.DataFrame(np.random.randn(2000,2)/10000, columns=['A','B'])
tmp['date'] = pd.date_range('2001-01-01',periods=2000)
tmp['ii'] = range(len(tmp))

def gm(ii, df, p):
    x_df = df.iloc[map(int, ii)]
    #print x_df
    v =((((x_df['A']+x_df['B'])+1).cumprod())-1)*p
    #print v
    return v.iloc[-1]

#print tmp.head()
res = pd.rolling_apply(tmp.ii, 50, lambda x: gm(x, tmp, 5))
print(res)



# conditions = {}
# series1 = pd.Series(np.random.randn(2000, 3),





