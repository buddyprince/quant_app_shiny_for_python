from __future__ import annotations
import pandas as pd
import numpy as np
import warnings
from typing import Callable
from IPython.display import display
import inspect
from typing import get_type_hints
from dataclasses import dataclass
from datetime import datetime
from matplotlib import pyplot as plt

class Time:
    def __init__(
        self,
        years: int = None,
        months: int = None,
        weeks: int = None,
        days: int = None,
        hours: int = None,
        minutes: int = None,
        seconds: int = None,
    ): 
        self._dict = dict(
            years =  years,
            months = months,
            weeks =  weeks,
            days =  days,
            hours =  hours,
            minutes = minutes,
            seconds = seconds
        )
    
    @property
    def values(self):
        return {k: v for k, v in self._dict.items() if v}

def _to_list(args: tuple|object):
    if isinstance(args, tuple):
        return list(args) 
    elif isinstance(args, list):
        return args
    else:
        return [args]


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 数据预处理 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def clean(df:pd.DataFrame, datetime_colume_label: str = 'date', freq: str = None):
    """
    清理源数据, 将行index改为DatetimeIndex
    Args:
        df(pd.DataFrame): 源数据dataframe
        datetime_colume_label(str): 源数据中包含时间信息的列label
        freq(str): 源数据时间频率
    """
    df[datetime_colume_label] = pd.to_datetime(df[datetime_colume_label], errors='coerce')
    df = df.set_index(datetime_colume_label).sort_index() 
    freq = df.index.to_series().diff().mean().floor("D") if freq is None else freq   # get the data frequency
    df = df.asfreq(freq)
    df = df.astype('float')
    return df

def _check_DatetimeIndex(df: pd.DataFrame|pd.Series):
    """
    确保DataFrame的行index的数据类型为pd.DatetimeIndex, 且已设定频率
    """
    if not isinstance(df, pd.DataFrame|pd.Series): 
        raise TypeError('信号数据类型不是DataFrame或Series')
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError('输入DateFrame行index的数据类型不是pd.DatetimeIndex')  
    if df.index.freq is None:
        raise TypeError('输入DateFrame行DatetimeIndex没有设定freq')        
    df.sort_index(inplace=True)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Domain Specific Language (DSL) $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
class DSL():
    """
    基于时间序列Pandas Dataframe的“领域特定语言 (domain specific langauge, DSL)”
    """
    def __init__(self, df:pd.DataFrame|pd.Series = None):
        if df is None:
            self._df = pd.DataFrame()
        self._df = pd.DataFrame(df.copy())
    
    # ---------------------------- Python 技术功能 ----------------------------
    def _to_df(self,x=None) -> pd.DataFrame:
        if x is None:
            out = self._df.squeeze("columns") if self._df.shape[1]==1 else self._df
        elif x is not None and isinstance(x, self.__class__):
            out = x._df.squeeze("columns") if self._df.shape[1]==1 else x._df 
        else:
            out = x
        return out
    
    def __getitem__(self, labels):
        """
        根据列label选择数据
        Args:
            labels: 要选择的列名
        """
        labels = _to_list(labels)
        temp = list(set(labels)-set(self._df.columns.to_list()))
        if len(temp)>0:
            raise KeyError(f"所选列labels不存在: {temp}")
        return self.__class__(self._df[labels])
    
    def __setitem__(self, labels, values):
        """
        根据列label赋值数据
        Args:
            labels: 要选择的列名
            values: 赋值
        """
        v = values._df if isinstance(values,DSL) else values
        self._df[_to_list(labels)] = v
        return self.__class__(self._df)
    
    @property
    def values(self):
        return self._df.copy()
    
    def describe(self):
        return self._df.describe()
    
    @property
    def dtypes(self):
        return self._df.dtypes

    # ---------------------------- 代数运算 ----------------------------
    def __add__(self, x):
        return self.__class__(self._to_df() + self._to_df(x))
    
    def __radd__(self, x):
        return self.__class__(self._to_df() + self._to_df(x))

    def __sub__(self, x):
        return self.__class__(self._to_df() - self._to_df(x))
    
    def __rsub__(self, x):
        return self.__class__(self._to_df() - self._to_df(x))

    def __mul__(self, x):
        return self.__class__(self._to_df() * self._to_df(x))
    
    def __rmul__(self, x):
        return self.__class__(self._to_df() * self._to_df(x))
        
    def __truediv__(self, x):
        return self.__class__(self._to_df() / self._to_df(x))
    
    def __rtruediv__(self, x):
        return self.__class__(self._to_df() / self._to_df(x))

    # ---------------------------- 比较 ----------------------------
    def __gt__(self, x): 
        return self.__class__(self._to_df() > self._to_df(x))
    
    def __ge__(self, x): 
        return self.__class__(self._to_df() >= self._to_df(x))
    
    def __lt__(self, x): 
        return self.__class__(self._to_df() < self._to_df(x))
        
    def __le__(self, x): 
        return self.__class__(self._to_df() <= self._to_df(x))
    
    def __eq__(self, x): 
        return self.__class__(self._to_df() == self._to_df(x))
    
    # ---------------------------- 逻辑运算 ----------------------------
    def __and__(self, x): 
        return self.__class__(self._to_df() & self._to_df(x))
    
    def __or__(self, x):  
        return self.__class__(self._to_df() | self._to_df(x))
    
    def __invert__(self):  
        return self.__class__(~self._to_df())


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 因子 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
class Factor(DSL):
    """
    因子构造, 一个时间序列Pandas DataFrame: 行index为DatetimeIndex, 列label为因子名称, 每一列为因子时间序列值:

    数据结构示意::
    
        +---------------+----------+----------+-----+
        | DatetimeIndex | factor 1 | factor 2 | ... |
        +===============+==========+==========+=====+
        | time          | value    | value    |     |
        +---------------+----------+----------+-----+
        | time          | value    | value    |     |
        +---------------+----------+----------+-----+
    
    """
    def __init__(self, df: pd.DataFrame):
        """
        因子构造, 一个时间序列Pandas DataFrame: 行index为DatetimeIndex, 列label为因子名称, 每一列为因子时间序列值:

        数据结构示意::

            +---------------+----------+----------+-----+
            | DatetimeIndex | factor 1 | factor 2 | ... |
            +===============+==========+==========+=====+
            | time          | value    | value    |     |
            +---------------+----------+----------+-----+
            | time          | value    | value    |     |
            +---------------+----------+----------+-----+

        Args:
            df(pd.DataFrame): 生成因子所基于的数据
        Examples:
            构造因子时，应按照以下方式定义, 
            注意最后一定要return f !!! ::

                def xxxxx(f: Factor):
                    pass
                    return f
            
        """
        _check_DatetimeIndex(df)
        super().__init__(df)

    # ---------------------------- Python 技术功能 ----------------------------
    def copy_DatetimeIndex(self):
        """
        生成一个全新的Factor类, 只保留DatetimeIndex
        """
        return self.__class__(pd.DataFrame(index = self._df.index.copy()))
        
    @property
    def _freq(self):
        return self._df.index.freq
    
    def _check_Time(self, window):
        if not isinstance(window, Time):
            raise TypeError('window的数据类型不是Time')

    # ---------------------------- 具体数据处理算法 ----------------------------
    def pct_change(
        self,
        window: Time,
        tolerance: Time = Time(days=7)       
    ):
        """
        环比变化
        Args:
            window(pd.DateOffset): 环比变化时间
            tolerance(Time): 最多允许偏差
        """
        self._check_Time(window)
        prev_df = self._df.reindex(
            self._df.index - pd.DateOffset(**window.values),
            method="ffill",
            tolerance=pd.Timedelta(**tolerance.values)
        ).set_index(self._df.index)

        return self.__class__(
            self._df / prev_df -1
        )
    
    def MA(self, window:Time):
        """
        滑动平均
        Args:
            window
        """
        self._check_Time(window)
        return self.__class__(
            self._df.rolling(pd.Timedelta(**window.values)).mean()
        )
    
    def MA_diff(self, window_1:Time, window_2:Time):
        """
        均线差
        """
        return self.__class__(
           self.MA(window_1)._df-self.MA(window_2)._df
        )
    
    def rolling_volatility(self, window:Time):
        """
        滚动波动率
        Args:
            window(int): the rolling window length
            unit(str): unit of windows
        """
        return self.__class__(
            self._df.rolling(pd.Timedelta(**window.values)).std()
        )


class FactorValues():
    """
    将“因子构造”, 结合具体数据，转换为具体因子值
        注意: 因子构造可以是相同的, 但是在不同交易品种数据下同一因子会有不同的因子值
    """
    def __init__(self, func: Callable[[Factor],Factor], df: pd.DataFrame):
        """
        将“因子构造”, 结合具体数据，转换为具体因子值
            注意: 因子构造可以是相同的, 但是在不同交易品种数据下同一因子会有不同的因子值
        Args:
            func(Callable): 因子构造
            df(DataFrame): 具体交易品种基础数据
        """
        self._check_factor_validity(func)
        _check_DatetimeIndex(df)
        self._func = func
        self._df = df
    
    def _check_factor_validity(self, func: Callable[[Factor],Factor]):
        """
        检查因子定义合法性
            1. 检查函数签名：必须一个参数
            2. 检查类型注解（强制写 xxx: Factor)
            3. 必须返回输入的参数自身
        Args:
            func: 因子构造
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        hints = get_type_hints(func, globalns=globals(), localns=locals())# get_type_hints 会解析 "Factor" 这类前向引用
        p0 = params[0].name
        self._p0 = p0
        if (
            len(params) != 1 or 
            p0 not in hints or 
            hints[p0] is not Factor
        ):
            raise TypeError(
                f"因子构造函数定义格式错误: def {func.__name__}{sig},应改为: \n"
                f"\t\u0020  def {self._func.__name__}({self._p0}: {Factor.__module__}.Factor) \n"
                f"\t\u0020\u0020\u0020\u0020  xxxx \n "
                f"\t\u0020\u0020\u0020\u0020  return {self._p0}  <--不要忘了这一行"
            )
        return func

    @property
    def values(self) -> pd.DataFrame:
        factor = Factor(self._df) # 1. 实例化一个factor类
        factor = self._func(factor) # 2. 函数在执行过程中，func中创造的因子会以一个新列的方式保存到实例化的factor._df中, 然后被返回
        try: 
            factor.values
        except AttributeError:
            raise TypeError(
                f"因子构造函数定义格式错误: 函数定义最后没有 return {self._p0}。根据当前符号，正确格式为: \n"
                f"\t\u0020  def {self._func.__name__}({self._p0}: {Factor.__module__}.Factor) \n"
                f"\t\u0020\u0020\u0020\u0020  xxxx \n "
                f"\t\u0020\u0020\u0020\u0020  return {self._p0}  <--不要忘了这一行"
            )
        factor = factor._df.drop(columns=self._df.columns) # 3. 在返回的factor._df中把原先self._df中的列去掉，剩下的就是新创造的因子
        return factor
    

class SingleFactorTesting():
    """
    单一因子测试引擎
        注意: 因子可以是相同的, 但是在不同交易品种(大豆、菜籽)下同一因子会有不同的因子值
    """
    def __init__(self, func: Callable[[Strategy],None], name: str = None):
        """
        单一因子测试引擎
            注意: 因子可以是相同的, 但是在不同交易品种(大豆、菜籽)下同一因子会有不同的因子值
        Args:
            func(Callable): 单一因子
            name(str): 因子名称
        """
        # 因子名称
        if name is not None:
            self._func_name = name
        elif name is None and inspect.getdoc(func) is not None:
            self._func_name = inspect.getdoc(func)
        else:
            self._func_name = func.__name__ 
        self._func = func # 单一因子构造
        self._market = pd.DataFrame() # 当前市场数据
        self._func_data = pd.DataFrame() # 当前因子绑定的数据
        self._all_values = pd.DataFrame()  # 单一因子生成的所有具体值
        self._all_markets = pd.DataFrame() # 全部市场价格数据

    def set_market_data(self, df:pd.DataFrame, name:str=None):
        """
        绑定单一交易品种的市场数据: 需要在市场数据中进行
        Args:
            df: 单一交易品种的市场数据
            name: 交易品种名称
        """
        _check_DatetimeIndex(df)
        if name is not None and name in df.columns.values:
            raise ValueError('交易品种名称重复')
        self._market = df
        self._market_name = name
        return self

    def _rename_by_market(self, df: pd.DataFrame):
        df.rename(columns={df.columns[-1]: self._market_name},inplace=True)

    def set_factor_data(self, df: pd.DataFrame|Factor):
        """
        绑定交易品种基础数据: 因子构造需要通过基础数据才能生成具体因子值
        Args:
            df: 基础数据
        """
        _check_DatetimeIndex(df)
        self._func_data = df
        return self

    def run(self):
        """
        开始按照因子构造生成具体因子值
        """
        if self._market.empty:
            raise ValueError('未绑定市场数据, 请先调用.set_market_data()绑定市场数据')
        if self._func_data.empty:
            raise ValueError('因子构造未绑定基础数据，无法转变为具体因子值, 请先调用.set_factor_data()绑定基础数据值')
        # 用self._market.index初始化Index, 保证全局索引对齐
        self._all_values = self._all_values.reindex(self._market.index.copy())
        self._all_markets = self._all_markets.reindex(self._market.index.copy()) 
        # 记录将当前因子具体值
        values = FactorValues(self._func, self._func_data).values
        self._all_values = self._all_values.join(values, rsuffix='_')
        self._rename_by_market(self._all_values)
        # 记录市场数据
        self._all_markets = self._all_markets.join(self._market, rsuffix='_')
        self._rename_by_market(self._all_markets)
        
        # 清空市场数据和基础数据值, 防止重复生成同一具体因子值
        self._market = pd.DataFrame()
        self._func_data = pd.DataFrame()
        return self
    
    @property
    def factors(self):
        return self._all_values
    
    @property
    def markets(self):
        return self._all_markets


class SingleFactorAnalysis:
    """
    单因子横截面分析
    """
    def __init__(self, factor_values_df:pd.DataFrame, markets_df:pd.DataFrame, name:str = None):
        """
        单因子横截面分析
        Args:
            factor_values_df(DataFrame): 单因子在不同交易品种下具体值的横截面数据
            markets_df(DataFrame): 不同交易品种的市场价格数据
            name: 因子名称
        """
        _check_DatetimeIndex(factor_values_df)
        _check_DatetimeIndex(markets_df)
        self._price_df = markets_df
        self._factor_values_df = factor_values_df
        self._forward_return_df = pd.DataFrame()
        self._func_name = name

    def set_horizon(self, horizon: Time = Time(days=1)):
        self._forward_return_df = self._price_df.shift(freq=-1*pd.Timedelta(**horizon.values))/self._price_df-1
        return self

    def _check_horizon(self):
        if self._forward_return_df.empty:
            raise ValueError('回报率计算跨度未设定，请先调用.set_horizon()设定回报率的跨度')

    @property
    def IC(self):
        """
        信息系数: t 时刻因子值与 t + horizon 时刻的收益率 (从 t 时刻开始计算) 之间的相关系数。
        Args:
            horizon(Time): 回报率的跨度
        """
        # 跨度为horizon的收益率
        self._check_horizon()
        return self._factor_values_df.corrwith(self._forward_return_df,axis=1,method='pearson').sort_index().rename('IC')
    
    @property
    def rankIC(self):
        self._check_horizon()
        return self._factor_values_df.corrwith(self._forward_return_df,axis=1,method='spearman').sort_index().rename('rankIC')
        
    
    def plot(self):
        """
        画图
        """
        plt.figure()
        ax = self.IC.plot(legend=True)
        ax.legend(loc='best')      
        ax.set_title(f'因子名称:{self._func_name}\n IC随时间变化记录')
        ax.set_xlabel('时间')
        ax.set_ylabel('IC')
        
        plt.figure()
        ax = self.rankIC.plot(legend=True)
        ax.legend(loc='best')      
        ax.set_title(f'因子名称:{self._func_name}\n rankIC随时间变化记录')
        ax.set_xlabel('时间')
        ax.set_ylabel('rankIC')



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 策略 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
class Strategy():
    """
    交易策略
    """
    def __init__(self, df: pd.DataFrame|Factor):
        """
        绑定因子值并初始化内部状态
        Args:
            df: 绑定的因子值
        Examples:
        
            自定义策略函数时，应按照以下方式定义::

                def xxxxx(xxx: Strategy):
                    pass
        """
        if isinstance(df, Factor):
            self._df = df.values
        elif isinstance(df, pd.DataFrame):
            self._df = df
        _check_DatetimeIndex(self._df)
        index = self._df.index
        self._conditions = pd.DataFrame(index=index) # 当前所有条件
        self._signals = pd.DataFrame(index=index) # 当前交易信号记录
        
    # ---------------------------- 因子选择 ----------------------------
    def select(self,*labels):
        """
        按labels选择因子
        Args:
            *labels(str|list): 待选择因子的labels
        """
        if self._df.empty:
            raise 
        return Factor(self._df)[labels]
    
    def __getitem__(self, labels):
        """
        按labels选择因子
        Args:
            *labels(str|list): 待选择因子的labels
        """
        return self.select(labels)
        
    # ---------------------------- 交易 ----------------------------
    def when(self, condition: Factor[bool]|pd.Series[bool]):
        """
        设定交易条件: 把条件绑定成向量化可执行规则, 并在其上触发交易动作(buy / sell /...) (作为condition → action的桥梁)。
        Args:
            condition(pd.Series[bool]): 交易条件, 一个 bool “mask”, 而非 if 控制流
        """
        ## 检查交易条件合法性
        # 1. 所有交易条件应是一个 bool “mask”
        col = condition._df if isinstance(condition, Factor) else condition
        col = pd.Series(col.squeeze("columns"))
        if not col.dtype == bool:
            raise TypeError('交易条件应该为bool变量')
        # 2. 在同一个策略下的同一个时间点，交易条件应唯一
        temp = pd.concat([self._conditions, col], axis=1)
        if ((temp.astype('int').sum(axis=1))>1).any():
            raise ValueError('交易条件冲突')
        ## 更新交易条件
        self._conditions = temp
        return self
    
    def buy(self, size:float|list[float]):
        """
        根据当前交易条件执行买入交易
        Args:
            size(float): 交易订单数量
        """
        self._signals.loc[self._conditions.iloc[:,-1],'size'] = size
        return self
    
    def sell(self, size:float|list[float]):
        """
        根据当前交易条件执行卖出交易
        Args:
            size(float): 交易订单数量
        """
        self._signals.loc[self._conditions.iloc[:,-1],'size'] = -size
        return self


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 信号 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
class StratagyValues():
    """
    策略具体值，即交易信号。 
        交易信号 = 策略 + 具体因子值: 将策略和具体因子值相结合，转变为交易信号 
        注意: 策略需要通过因子值才能生成交易信号, 同一策略在不同数据下会产生不同的交易信号
    """
    def __init__(self,func: Callable[[Strategy],None], df: pd.DataFrame|Factor):
        """
        Args:
            func(Callable): 自定义策略
            df(DataFrame): 因子值
        """
        self._check_strategy_validity(func)
        _check_DatetimeIndex(df)
        self._func = func
        self._df = df
        
    def _check_strategy_validity(self, func: Callable[[Strategy],None]):
        """
        检查策略定义合法性
            1. 检查函数签名：必须一个参数
            2. 检查类型注解（强制写 xxx: Strategy)
        Args:
            func: 自定义策略
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        hints = get_type_hints(func, globalns=globals(), localns=locals())# get_type_hints 会解析 "Strategy" 这类前向引用
        p0 = params[0].name
        if (
            len(params) != 1 or 
            p0 not in hints or 
            hints[p0] is not Strategy
        ):
            raise TypeError(
                f"策略函数定义格式错误: def {func.__name__}{sig}, "
                f"应改为: def {func.__name__}({p0}: {Strategy.__module__}.Strategy)"  
            )
        return func

    @property
    def values(self) -> pd.DataFrame:
        _check_DatetimeIndex(self._df)
        # strategy的本质是._condition和._signals
        strategy = Strategy(self._df) # 1. 通过实例化一个strategy类来初始化._condition和._signals
        self._func(strategy) # 2. 函数在执行过程中，strategy里面的._condition和._signals会被func中调用的when、buy、sell改写
        signals = strategy._signals # 3. 在运行结束后，再次调用self._strategy._condition和self._strategy._signals就是我们要的执行结果
        # 将所有没有交易信号的地方填充0
        return signals.fillna(0) 


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 回测 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 注意：回测的对象是策略，而不是因子。策略可以认为是由因子组成的。回测的好坏直接说明策略的表现，而策略的表现会收到因子有效性的影响。因子有自己的评价指标（例如IC、IR）
class SingleStrategyBackTesting():
    """
    单一策略交易
        注意: 策略不等于交易品种(大豆、菜籽), 单一策略可以应用到多种交易品种生成不同交易信号
    """
    def __init__(self, func: Callable[[Strategy],None], name: str = None):
        """
        单一策略交易
            注意: 策略不等于交易品种(大豆、菜籽), 单一策略可以应用到多种交易品种生成不同交易信号
        Args:
            func(Callable): 单一策略
            name(str): 策略名称
        """
        # 策略名称
        if name is not None:
            self._func_name = name
        elif name is None and inspect.getdoc(func) is not None:
            self._func_name = inspect.getdoc(func)
        else:
            self._func_name = func.__name__ 
        self._func = func # 单一策略
        self._market = pd.DataFrame() # 当前市场数据
        self._func_data = pd.DataFrame() # 当前策略绑定数据
        self._all_signals = pd.DataFrame()  # 单一策略生成的所有信号
        self._all_holding_positions = pd.DataFrame() # 持仓记录
        self._all_cash_flows = pd.DataFrame() # 所有现金流变动记录

    def set_market_data(self, df:pd.DataFrame, name:str=None):
        """
        绑定单一交易品种的市场数据: 交易需要在市场数据中进行
        Args:
            df: 单一交易品种的市场数据
            name: 交易品种名称
        """
        _check_DatetimeIndex(df)
        if name is not None and name in df.columns.values:
            raise ValueError('交易品种名称重复')
        self._market = df
        self._market_name = name
        return self

    def _rename_by_market(self, df: pd.DataFrame):
        df.rename(columns={df.columns[-1]: self._market_name},inplace=True)

    def set_strategy_data(self, df: pd.DataFrame|Factor):
        """
        绑定因子值: 策略需要通过因子值才能生成交易信号
        Args:
            df: 因子值
        """
        _check_DatetimeIndex(df)
        self._func_data = df
        return self

    def _signals_to_positions(self, signals:pd.DataFrame) -> pd.DataFrame:
        """
        由信号合成持仓量
        """
        return signals.cumsum()

    def run(self):
        """
        开始按照该策略进行交易
        """
        if self._market.empty:
            raise ValueError('未绑定市场数据，无法交易, 请先调用.set_market_data()绑定市场数据')
        if self._func_data.empty:
            raise ValueError('未绑定因子值，策略无法转变为信号, 请先调用.set_strategy_data()绑定因子值')
        # 用self._market.index初始化Index, 保证全局索引对齐
        self._all_signals = self._all_signals.reindex(self._market.index.copy())
        self._all_holding_positions = self._all_holding_positions.reindex(self._market.index.copy()) 
        self._all_cash_flows = self._all_cash_flows.reindex(self._market.index.copy()) 
        # 记录将当前信号
        signals = StratagyValues(self._func, self._func_data).values
        self._all_signals = self._all_signals.join(signals, rsuffix='_')
        self._rename_by_market(self._all_signals)
        # 将信号合成持仓量
        positions = self._signals_to_positions(signals)
        self._all_holding_positions = self._all_holding_positions.join(positions, rsuffix='_')
        self._rename_by_market(self._all_holding_positions)
        # 通过逐日盯市计算盈亏
        price_diff = self._market['price'].diff() # 逐日盯市的价差
        self._all_cash_flows = self._all_cash_flows.join(positions.shift(periods=1).mul(price_diff, axis=0), rsuffix='_')
        self._rename_by_market(self._all_cash_flows)
        # 清空市场数据和信号, 防止重复交易同一交易品种
        self._market = pd.DataFrame()
        self._func_data = pd.DataFrame()
        return self
    
    @property
    def cumulative_profit_and_loss(self) -> pd.DataFrame:
        """
        计算累计盈亏
        """
        return self._all_cash_flows.resample(pd.Timedelta(days=1)).sum().cumsum()

    def plot(self):
        """
        画图
        """
        ax = self.cumulative_profit_and_loss.plot(legend=True)
        ax.legend(loc='best')      
        ax.set_title(f'策略名称:{self._func_name}\n 累计盈亏随时间变化记录')
        ax.set_xlabel('时间')
        ax.set_ylabel('货币单位')
        
        ax = self._all_holding_positions.plot(legend=True)
        ax.legend(loc='best')      
        ax.set_title(f'策略名称:{self._func_name}\n 持仓量随时间变化记录')
        ax.set_xlabel('时间')
        ax.set_ylabel('仓位')


class SingleStatagyAnalysis():
    """
    单一策略量化结果分析
    """
    def __init__(self, equity_curve: pd.DataFrame):
        """
        单一策略量化结果分析

        Args:
            equity_curve: 单一策略在不用交易品种下的权益曲线累计盈亏
        
        数据结构示意::

            +---------------+----------+------------+-----+
            | DatetimeIndex | stock 1  | comodity 2 | ... |
            +===============+==========+============+=====+
            | time          | value    | value      |     |
            +---------------+----------+------------+-----+
            | time          | value    | value      |     |
            +---------------+----------+------------+-----+
        """
        _check_DatetimeIndex(equity_curve)
        if equity_curve.index.freq == 'D':
            equity_curve.resample(pd.Timedelta(days=1)).last()
        total_days = equity_curve.shape[0] # 总天数
        K = 252 # 年交易日
        risk_free_rate = 0.02
        
        # 单日收益率
        daily_rate_of_return = (
            equity_curve
            .groupby(pd.Grouper(freq='D')).sum()
            .pct_change()
        ) 
        # 累计收益
        final_return = equity_curve.iloc[-1]/equity_curve.iloc[0] - 1
        # 年化收益
        annualized_percentage_rate = (K*final_return/total_days)-1
        # 年化波动率
        annulized_volatility = equity_curve.std()*np.sqrt(K)
        # Sharpe Ratio
        shape_ratio = (annualized_percentage_rate-risk_free_rate)/annulized_volatility
        # 最大回撤
        max_drawdown = (equity_curve/equity_curve.cummax()-1).min()
        # 胜率
        win_rate = (daily_rate_of_return>0).sum()/total_days
        
        self._out = pd.concat([
            final_return.rename("累计收益率"),
            annualized_percentage_rate.rename("年化收益(APR,不是EAR)"),
            annulized_volatility.rename("年化波动率"),
            shape_ratio.rename("Sharpe Ratio"),
            max_drawdown.rename("最大回撤"),
            win_rate.rename("胜率"),
        ],axis=1)

    @property
    def values(self):
        return self._out.round(1)
    
    def display(self):
        display(self._out.round(1).astype(str) + '%')
