
from pathlib import Path
import pandas as pd
from shiny import reactive
from shiny.express import input, render, ui
from shiny.ui import output_ui
from shiny.types import FileInfo
from typing import Callable, get_type_hints
import inspect
import sys
from datetime import date
ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 字体路径（相对项目根目录）
font_path = Path(__file__).parent/"SourceHanSansSC-Regular.otf"
# 注册字体
fm.fontManager.addfont(str(font_path))
prop = fm.FontProperties(fname=str(font_path))

# 全局设置
plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False   # 解决负号显示问题


import quant
import server


current_file_path = Path(__file__).resolve()
current_dir = Path(__file__).resolve().parent

# Default to the last 6 months
end = pd.Timestamp.now()
start = end - pd.Timedelta(weeks=26)

market_data_dict = reactive.Value(None)
factor_data_dict = reactive.Value(None)
factor_func_dict = reactive.Value(None)   # 保存解析后的函数信息
strategy_func_dict = reactive.Value(None)   # 保存解析后的函数信息
factor_func_selected = reactive.Value(None)
strategy_func_selected = reactive.Value(None)
trading_instrument_list = reactive.Value(None)
@reactive.effect
def update_trading_instrument_list():
    """
    交易品种需要既有市场数据, 也有因子构建基础数据, 否则无法得到具体数值进行分析。
    """
    if factor_data_dict() and market_data_dict():
        trading_instrument_list.set(list(factor_data_dict().keys()&market_data_dict().keys()))

trading_instrument_selected = reactive.Value(None)

ui.page_opts(title="量化", window_title = '量化')
ui.head_content(
    ui.include_css(f"{current_dir}/style.css")
)
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 说明 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
with ui.nav_panel("说明"):
    with ui.card(fill=True):
        ui.card_header('数据说明')
        ui.div('数据分为两种, <市场价格数据>和<因子构造基础数据>，要求这两种数据在上传时命名一样。')
        ui.div(
            '这两种数据缺一不可，因为没有<市场价格数据>就无法进行横截面分析和回测，没有<因子构造基础数据>因子就无法生成具体值' 
        )
        ui.div(
            '举个例子：我们现在有大豆的<CBOT连续合约的市场价格数据>和<大连期货交易所的持仓数据>,'\
            '我们想用持仓数据构建一个“持仓波动率”因子，然后测试这个因子的表现（也就是横截面分析）。'\
            '那么此时<CBOT连续合约的市场价格数据>就是<市场价格数据>，<大连期货交易所的持仓数据>就是<因子构造基础数据>。'
        )
        ui.div(
            '那么我们在上传<市场价格数据>时就需要把<CBOT连续合约的市场价格数据>命名为<大豆.csv>,' \
            '在上传<因子生成基础数据>时也需要把<大连期货交易所的持仓数据>命名为<大豆.csv>。'
        )
        ui.div(
            '不用担心重名，因为<数据上传>页面已经把这两种数据分开上传了。' 
        )

    with ui.card(fill=True):
        ui.card_header('概念说明')
        ui.div('因子和策略，分为构造和具体值，千万不能搞混。')
        ui.div(
            '同一个因子构造，在不同交易品种数据下会有不同的因子值。比如我们构造了一个因子叫<价格动量>，那么具体到大豆还是菜籽，' \
            '会得到完全不同的因子具体值，但是他们都在用共同的一个因子构造叫<价格动量>' \
            '策略同理，同一个策略构造，在不同市场环境下会产生完全不同的策略具体值（或者换一个熟悉的名字叫做交易信号）。'
        )
        ui.div(
            '因子构造需要<因子构造基础数据>生成具体值，然后在<市场价格数据>下进行横截面分析。' \
            '策略构造需要结合因子具体值生成策略具体值（交易信号），然后在<市场价格数据>下进行回测。' \
            '横截面分析和回测的主体千万不能搞混。'
        )
        
    


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 数据上传 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
with ui.nav_panel("数据上传"):
    with ui.card(fill=True):
        ui.card_header('市场价格数据')
        ui.input_file(
            id='input_market_data', label=ui.p("市场价格数据"), 
            accept=[".csv"], width = '100%', button_label='浏览文件', 
            placeholder='请上传一个或多个.csv文件(其中包含市场数据)', multiple=True
        )
        @reactive.effect
        def get_market_data():
            market_data_dict.set(
                server.reactive_effect_get_data_dict(input.input_market_data())
            )
        @render.ui
        def market_header():
            return server.render_ui_upload_header(market_data_dict())
        @reactive.calc
        def market_describe():
            return server.reactive_calc_describe_df(market_data_dict())
        @render.ui
        def market_preview_df():
            return server.render_ui_upload_preview_df(market_describe())


    with ui.card(fill=True):
        ui.card_header('因子生成基础数据')
        ui.input_file(
            id='input_factor_data', label=ui.p("因子生成基础数据"), 
            accept=[".csv"],width = '100%', button_label='浏览文件', 
            placeholder='请上传一个或多个.csv文件(其中包含因子生成基础数据)', multiple=True
        )
        @reactive.effect
        def get_factor_data():
            factor_data_dict.set(
                server.reactive_effect_get_data_dict(input.input_factor_data())
            )
        @render.ui
        def factor_data_header():
            return server.render_ui_upload_header(factor_data_dict())
        @reactive.calc
        def factor_data_describe():
            return server.reactive_calc_describe_df(factor_data_dict())
        @render.ui
        def factor_preview_df():
            return server.render_ui_upload_preview_df(factor_data_describe())


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 因子 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
with ui.nav_panel("因子构造"):
    with ui.card(fill=True):
        ui.card_header('因子构造')
        ui.input_file(
            id='factor_construction_from_file', label=ui.p("因子构造方式"), 
            accept=[".py,.txt"],width = '100%', button_label='浏览文件', 
            placeholder='请上传一个 .py/.txt 文件（其中定义函数）', multiple=False
        )
        @reactive.effect
        def load_factor_func_from_file():
            """
            读取因子构造函数
            """
            factor_func_dict.set(
                server.reactive_calc_load_functions_from_file(input.factor_construction_from_file(), quant.Factor)
            )
        @render.ui
        def factor_consturction_preview_func():
            return server.render_ui_upload_preview_func(factor_func_dict())


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 策略 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
with ui.nav_panel("策略构造"):
    with ui.card(fill=True):
        ui.card_header('策略构造')
        ui.input_file(
            id='strategy_construction_from_file', label=ui.p("策略构造方式"), 
            accept=[".py,.txt"],width = '100%', button_label='浏览文件', 
            placeholder='请上传一个 .py/.txt 文件（其中定义函数）', multiple=False
        )
        @reactive.calc
        def load_strategy_func_from_file(): 
            """
            读取策略构造函数
            """
            strategy_func_dict.set(
                server.reactive_calc_load_functions_from_file(input.strategy_construction_from_file(), quant.Strategy)
            )
            return strategy_func_dict()
        @render.ui
        def strategy_consturction_preview_func():
            return server.render_ui_upload_preview_func(load_strategy_func_from_file())


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 单因子横截面分析 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
with ui.nav_panel("单因子横截面分析"):
    with ui.card():
        ui.card_header('参数设定')
        # ----------------------因子构造----------------------
        ui.input_select(
            id = "select_factor_construction_func_SingleFactorAnalysis", label = "选择一个因子构造函数",
            choices=[],  # 初始空，后面动态更新
            selected=None, width='100%'
        )
        @reactive.effect
        def update_factor_choices_SingleFactorAnalysis():
            if not factor_func_dict():
                ui.update_select(
                    id="select_factor_construction_func_SingleFactorAnalysis", label='请选择一个因子',
                    choices={}, selected=None
                )
                return 
            # choices 用 “函数名” 作为 key；显示文字可带 signature
            choices = {func.__name__: f"{func.__name__} {str(inspect.signature(func))}" for func in factor_func_dict().values()}
            ui.update_select(
                id="select_factor_construction_func_SingleFactorAnalysis", label='请选择一个因子',
                choices=choices, selected=None
            )
        @reactive.effect
        def get_factor_func_selected_SingleFactorAnalysis():
            factor_func_selected.set(input.select_factor_construction_func_SingleFactorAnalysis())
        @render.text
        def print_factor_func_selected_SingleFactorAnalysis():
            if not factor_func_selected():
                return f"未选择"
            return f"已选择: {factor_func_selected()}"

        # ----------------------交易品种数据----------------------
        ui.div(
            ui.input_selectize(
                id = "select_trading_instrument_SingleFactorAnalysis", label = "选择一个或多个交易品种",
                choices=[],  # 初始空，后面动态更新
                selected=None, 
                multiple=True,
                width = '100%',
            )
        )
        @reactive.effect
        def update_trading_instrument_choices_SingleFactorAnalysis():
            if not trading_instrument_list():
                ui.update_select(
                    id="select_trading_instrument_SingleFactorAnalysis",
                    choices={}, selected=None
                )
                return 
            # choices 用 “函数名” 作为 key；显示文字可带 signature
            choices = [name for name in trading_instrument_list()]
            ui.update_select(
                id="select_trading_instrument_SingleFactorAnalysis", 
                choices=choices, 
                selected=choices
            )
        @reactive.effect
        def get_trading_instrument_selected_SingleFactorAnalysis():
            trading_instrument_selected.set(input.select_trading_instrument_SingleFactorAnalysis())
        @render.text
        def print_trading_instrument_selected_SingleFactorAnalysis():
            if not trading_instrument_selected():
                return f"未选择"
            return f"已选择: {trading_instrument_selected()}"

        # ----------------------日期范围---------------------
        ui.input_date_range(id='single_factor_testing_date_range',
            label='横截面分析日期范围',start='2000-01-01',end=None, startview='year'
        )

        # ----------------------开始测试按钮----------------------
        ui.input_task_button(
            id='run_single_factor_testing_button',label='开始测试',
            label_busy='计算中',width='100%'
        )

    # ----------------------横截面分析----------------------
    # 定义extended_task
    @ui.bind_task_button(button_id="run_single_factor_testing_button")
    @reactive.extended_task
    async def run_single_factor_testing(
        factor_func: Callable, factor_data_dict: dict, market_data_dict: dict,
        start:date, end:date
    ): 
        try:
            return server.reactive_extended_task_SingleFactorTesting_pipeline(
                factor_func, factor_data_dict, market_data_dict, start, end
            )
        except Exception as e:
            return e
        
    # 点击按钮，开始运行extended_task
    @reactive.effect
    @reactive.event(input.run_single_factor_testing_button, ignore_none=True)
    def handle_click_SingleFactorAnalysis():
        start, end = input.single_factor_testing_date_range()
        try:
            run_single_factor_testing(
                factor_func_dict()[factor_func_selected()],
                {k: factor_data_dict()[k] for k in quant._to_list(trading_instrument_selected())},
                {k: market_data_dict()[k] for k in quant._to_list(trading_instrument_selected())},
                start, end
            )
        except Exception as e:
            run_single_factor_testing.cancel()
            print(e)

    with ui.card():
        ui.card_header("状态")
        @render.text
        def print_SingleFactorAnalysis():
            if not factor_func_selected() or not trading_instrument_selected():
                return '未选择交易品种或因子'
            results = run_single_factor_testing.result()
            if isinstance(results, Exception):
                return results
            return f'以下横截面分析，基于{len(trading_instrument_selected())}种不同交易品种'
    
    with ui.card():
        ui.card_header("横截面分析")
        @render.data_frame
        def print_IC():
            results = run_single_factor_testing.result()
            df = pd.DataFrame()
            df['IC']=results.IC.round(1)
            df['rankIC'] = results.rankIC.round(1)
            return df.reset_index()
        
        
        @render.plot
        def plot_IC():
            if not run_single_factor_testing.result():
                return None
            results = run_single_factor_testing.result()
            MA = 60
            ax = results.IC.rolling(pd.Timedelta(days=MA)).mean().plot()
            ax.legend(loc='best')      
            ax.set_title(f'因子名称:{results._func_name}\n IC随时间变化记录, 滑动平均_{MA}天')
            ax.set_xlabel('时间')
            ax.set_ylabel('IC')
            return ax
        @render.plot
        def plot_rankIC():
            if not run_single_factor_testing.result():
                return None
            results = run_single_factor_testing.result()
            MA = 60
            ax = results.rankIC.rolling(pd.Timedelta(days=MA)).mean().plot()
            ax.legend(loc='best')      
            ax.set_title(f'因子名称:{results._func_name}\n rankIC随时间变化记录，滑动平均_{MA}天')
            ax.set_xlabel('时间')
            ax.set_ylabel('rankIC')
            return ax


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 单策略回测分析 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
with ui.nav_panel("单策略回测分析"):
    with ui.card():
        ui.card_header('参数设定')
        # ----------------------策略构造----------------------
        ui.input_select(
            id = "select_strategy_construction_func_SingleStrategyBackTesting", label = "选择一个策略构造函数",
            choices=[],  # 初始空，后面动态更新
            selected=None, width='100%'
        )
        @reactive.effect
        def update_strategy_choices_SingleStrategyBackTesting():
            if not strategy_func_dict():
                ui.update_select(
                    id="select_strategy_construction_func_SingleStrategyBackTesting", label='请选择一个策略',
                    choices={}, selected=None
                )
                return 
            # choices 用 “函数名” 作为 key；显示文字可带 signature
            choices = {func.__name__: f"{func.__name__} {str(inspect.signature(func))}" for func in strategy_func_dict().values()}
            ui.update_select(
                id="select_strategy_construction_func_SingleStrategyBackTesting", label='请选择一个策略',
                choices=choices, selected=None
            )
        @reactive.effect
        def get_strategy_func_selected_SingleStrategyBackTesting():
            strategy_func_selected.set(input.select_strategy_construction_func_SingleStrategyBackTesting())
        @render.text
        def print_factor_func_selected_SingleStrategyBackTesting():
            if not strategy_func_selected():
                return f"未选择"
            return f"已选择: {strategy_func_selected()}"
    
    
        @render.ui
        def available_factors_SingleStrategyBackTesting():
            header_1 = ui.div(ui.span(f"请确认: 所选策略中涉及的全部因子已经在“因子构造”页面中上传了构造函数", class_="func-label"))
            header_2 = server.render_ui_upload_preview_func(factor_func_dict(),header_only=True)
            if not header_2:
                return ui.div(
                    header_1,  
                    ui.div(ui.span('无可用因子构造函数', class_="func-label"))
                )
            return ui.div(header_1, header_2)

        # ----------------------交易品种数据----------------------
        ui.div(
            ui.input_selectize(
                id = "select_trading_instrument_SingleStratagyBackTesting_pipeline", label = "选择一个或多个交易品种",
                choices=[],  # 初始空，后面动态更新
                selected=None, 
                multiple=True,
                width = '100%',
            )
        )
        @reactive.effect
        def update_trading_instrument_choices_SingleStratagyBackTesting_pipeline():
            if not trading_instrument_list():
                ui.update_select(
                    id="select_trading_instrument_SingleStratagyBackTesting_pipeline",
                    choices={}, selected=None
                )
                return 
            # choices 用 “函数名” 作为 key；显示文字可带 signature
            choices = [name for name in trading_instrument_list()]
            ui.update_select(
                id="select_trading_instrument_SingleStratagyBackTesting_pipeline", 
                choices=choices, 
                selected=choices
            )
        @reactive.effect
        def get_trading_instrument_selected_SingleStratagyBackTesting_pipeline():
            trading_instrument_selected.set(input.select_trading_instrument_SingleStratagyBackTesting_pipeline())
        @render.text
        def print_trading_instrument_selected_SingleStratagyBackTesting_pipeline():
            if not trading_instrument_selected():
                return f"未选择"
            return f"已选择: {trading_instrument_selected()}"

        # ----------------------回测参数设定----------------------

        ui.input_numeric(id="initial_fund", label="初始资金:", value=1000000, min=1, update_on='blur')
        ui.input_date_range(id='single_stratagy_backtesting_date_range',
            label='回测日期范围',start='2000-01-01',end=None, startview='year'
        )
        # ----------------------开始测试按钮----------------------
        ui.input_task_button(
            id='run_single_stratagy_backtesting_button',label='开始测试',
            label_busy='计算中',width='100%'
        )

    # ----------------------回测----------------------
    # 定义extended_task
    @ui.bind_task_button(button_id="run_single_stratagy_backtesting_button")
    @reactive.extended_task
    async def run_single_stategy_back_testing(
        stratagy_func: Callable, factor_func_list:list[Callable], factor_data_dict: dict, market_data_dict: dict,
        initial_fund: float, start:date, end:date
    ): 
        try:
            return server.reactive_extended_task_SingleStratagyBackTesting_pipeline(
                stratagy_func, factor_func_list, factor_data_dict, market_data_dict, 
                initial_fund, start, end
            )
        except Exception as e:
            return e
    # 点击按钮，开始运行extended_task
    @reactive.effect
    @reactive.event(input.run_single_stratagy_backtesting_button, ignore_none=True)
    def handle_click_SingleStratagyBackTesting():
        start,end = input.single_stratagy_backtesting_date_range()
        try:
            run_single_stategy_back_testing(
                strategy_func_dict()[strategy_func_selected()],
                list(factor_func_dict().values()),
                {k: factor_data_dict()[k] for k in quant._to_list(trading_instrument_selected())},
                {k: market_data_dict()[k] for k in quant._to_list(trading_instrument_selected())},
                input.initial_fund(), start, end

            )
        except Exception as e:
            run_single_stategy_back_testing.cancel()
            print(e)
    
    with ui.card():
        ui.card_header("状态")
        @render.text
        def print_SingleStratagyBackTesting():
            if not strategy_func_selected() or not trading_instrument_selected():
                return '未选择交易品种或因子'
            results = run_single_stategy_back_testing.result()
            if isinstance(results, Exception):
                return results
            return f'以下是{len(trading_instrument_selected())}种不同交易品种的回测结果'
    
    with ui.card():
        ui.card_header("回测结果")
        
        @render.data_frame
        def print_SingleStatagyAnalysis():
            ssbt, df = run_single_stategy_back_testing.result()
            return df.reset_index()
            

        @render.plot
        def plot_cumulative_profit_and_loss():
            if not run_single_stategy_back_testing.result():
                return None
            ssbt, df = run_single_stategy_back_testing.result()
            ax = (ssbt.cumulative_profit_and_loss+input.initial_fund()).plot(legend=True)
            ax.legend(loc='best')      
            ax.set_title(f'策略名称:{ssbt._func_name}\n 权益曲线随时间变化记录')
            ax.set_xlabel('时间')
            ax.set_ylabel('货币单位')
            return ax
        
        @render.plot
        def plot_all_holding_positions():
            if not run_single_stategy_back_testing.result():
                return None
            ssbt, df = run_single_stategy_back_testing.result()
            ax = ssbt._all_holding_positions.plot(legend=True)
            ax.legend(loc='best')      
            ax.set_title(f'策略名称:{ssbt._func_name}\n 持仓量随时间变化记录')
            ax.set_xlabel('时间')
            ax.set_ylabel('仓位')
            return ax
