import quant

def factor_test_1(f: quant.Factor):
    """收益率滚动波动率1"""
    period_rate_of_return = f['price'].pct_change(quant.Time(weeks=1))
    f['收益率滚动波动率'] = period_rate_of_return.rolling_volatility(quant.Time(weeks=1))
    return f

def factor_test_2(f: quant.Factor):
    """收益率滚动波动率2"""
    period_rate_of_return = f['price'].pct_change(quant.Time(weeks=2))
    f['收益率滚动波动率'] = period_rate_of_return.rolling_volatility(quant.Time(weeks=1))
    return f

def stratagy_test_1(s: quant.Strategy):
    """策略1"""
    s.when(s['收益率滚动波动率']>0.005).sell(1)
    s.when(s['收益率滚动波动率']<0.004).buy(2)
