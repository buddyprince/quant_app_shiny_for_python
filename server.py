from pathlib import Path
import pandas as pd
import numpy as np
import types
from typing import Callable, get_type_hints
import inspect
from shiny import reactive
from shiny.express import input, render, ui
from shiny.ui import output_ui
from shiny.types import FileInfo
from datetime import date
import sys
ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(ROOT))
import quant

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 数据上传 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def reactive_effect_get_data_dict(input_FileInfo_list: list[FileInfo] | None):
    if not input_FileInfo_list:     # 同时处理 None 和 []
        return {}
    data_dict = {file['name']: file['datapath'] for file in input_FileInfo_list}
    return data_dict

def reactive_calc_describe_df(data_dict: dict[str, pd.DataFrame] | None):
    out = {}
    if data_dict is not None:
        for name in data_dict.keys():
            df = quant.clean(pd.read_csv(data_dict[name])).reset_index().describe().round(1)
            out[name] = df
    return out        

def render_ui_upload_header(data_dict: dict[str, pd.DataFrame] | None):
    if not data_dict:
        return ui.p("请上传文件")
    filenames = [name for name in data_dict.keys()]
    header = ui.div(
        ui.span(f"已上传{len(filenames)}个文件", class_="upload-label"),
        *[ui.span(name, class_="upload-chip") for name in filenames],
        class_="upload-header"
    )
    return header

def render_ui_upload_preview_df(df_dict: dict[str, pd.DataFrame]):
    if not df_dict:
        return None
    items = []
    for name, desc in df_dict.items():
        items.append(
            ui.tags.details(
                ui.tags.summary(name),
                ui.div(
                    ui.HTML(desc.to_html(border=0)),
                    class_="desc-card",
                ),
                class_="acc-item",
                open=False,  # 默认收起；想默认展开就 True
            )
        )
    inner = ui.div(*items, class_="acc")
    outer = ui.tags.details(
        ui.tags.summary(f"文件预览（{len(df_dict)} 个）"),
        inner,
        class_="acc-item",
        open=False 
    )
    return ui.div(outer)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 因子+策略构造 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def reactive_calc_load_functions_from_file(input_FileInfo_list: list[FileInfo] | None, type: object):
    if input_FileInfo_list is None:
        return None
    py_path = input_FileInfo_list[0]["datapath"]
    namespace = {"quant": quant}
    with open(py_path, "r", encoding="utf-8") as f:
        code = f.read()
    try:
        compiled = compile(code, py_path, "exec")
        exec(compiled, namespace)
    except ModuleNotFoundError as e:
        raise RuntimeError(f"缺少依赖包：{e.name}。请先在运行环境安装。")
    
    func_dict = {}
    for name, obj in namespace.items():
        # 只要“普通 def 定义的函数”
        if isinstance(obj, types.FunctionType):
            # 确保要是合法定义
            sig = inspect.signature(obj)
            params = list(sig.parameters.values())
            hints = get_type_hints(obj, globalns=globals(), localns=locals()) # get_type_hints 会解析 "Factor" 这类前向引用
            p0 = params[0].name
            if (
                len(params) != 1 or 
                p0 not in hints or 
                hints[p0] is not type
            ): 
                continue
            func_dict[obj.__name__] = obj
    return func_dict


def render_ui_upload_preview_func(func_dict: dict[str,Callable], header_only = False):
    if not func_dict:
        return None
    
    header = ui.div(
        ui.span(f"已导入的构造函数", class_="func-label"),
        *[
            ui.div(
                ui.span(
                    f"{func.__name__}{str(inspect.signature(func))}",
                    class_="func-chip"
                )
            )
            for func in func_dict.values()
        ],
        class_="func-header"
    )

    items = []
    for func in func_dict.values():
        items.append(
            ui.tags.details(
                ui.tags.summary(
                    ui.span(
                        f"{func.__name__}{str(inspect.signature(func))}",
                        class_="func-chip"
                    )
                ),
                ui.div(
                    ui.pre(
                        ui.code(inspect.getsource(func)),
                        class_="func-source"
                    ),
                    class_="func-item"
                ),
                class_="acc-item",
                open=False,  # 默认收起；想默认展开就 True
            )
        )
    inner = ui.div(*items, class_="acc")
    outer = ui.tags.details(
        ui.tags.summary(f"构造函数预览（{len(func_dict)} 个）"),
        inner,
        class_="acc-item",
        open=False 
    )
    if header_only:
        return header
    else: 
        return ui.div(header,outer)



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 因子横截面分析 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def reactive_extended_task_SingleFactorTesting_pipeline(
        factor_func: Callable, factor_data_dict: dict, market_data_dict: dict,
        start:date, end:date
    ):
    if not factor_data_dict.keys() == market_data_dict.keys(): 
        raise KeyError('交易品种的市场数据与因子基础数据不一致')
    # 1. 初始化单一因子测试引擎
    sft = quant.SingleFactorTesting(factor_func) 
    for name in factor_data_dict.keys():
        # 2. 输入因子数据和市场数据
        df = quant.clean(pd.read_csv(market_data_dict[name]), datetime_colume_label='date')
        df = df.loc[start:end]
        sft.set_market_data(df,name)
        df = quant.clean(pd.read_csv(factor_data_dict[name]), datetime_colume_label='date')
        df = df.loc[start:end]
        sft.set_factor_data(df)
        # 3. 对该交易品种进行单因子测试
        sft.run()
    # 4. 所有测试结束后，进行结果评估
    sfa = quant.SingleFactorAnalysis(sft.factors,sft.markets,inspect.getdoc(factor_func)).set_horizon(quant.Time(days=1))
    return sfa

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 单策略回测分析 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def reactive_extended_task_SingleStratagyBackTesting_pipeline(
        stratagy_func: Callable, factor_func_list:list[Callable], factor_data_dict: dict, market_data_dict: dict,
        initial_fund:float, # 初始资金
        start: date, # 开始时间
        end:date # 结束时间
    ):
    # 1. 初始化回测引擎
    ssbt = quant.SingleStrategyBackTesting(stratagy_func) 
    for name in factor_data_dict.keys():
        # 3. 计算在不同种类期货上的具体因子值
        factor = pd.DataFrame()
        for factor_func in factor_func_list:
            df = quant.clean(pd.read_csv(factor_data_dict[name]), datetime_colume_label='date')
            df = df.loc[start:end]
            factor = factor.join(
                quant.FactorValues(factor_func, df).values,
                rsuffix='_',how = 'outer'
            )
        # 3. 输入因子数据和市场数据
        df = quant.clean(pd.read_csv(market_data_dict[name]), datetime_colume_label='date')
        df = df.loc[start:end]
        ssbt.set_market_data(
            df, 
            name= name
        )
        ssbt.set_strategy_data(factor)
        
        # 4. 计算策略值（信号值），对该期货进行回测交易
        ssbt.run()
    equity_curve = ssbt.cumulative_profit_and_loss + initial_fund # 由累计盈亏生成权益曲线
    df = quant.SingleStatagyAnalysis(equity_curve).values
    return ssbt, df

