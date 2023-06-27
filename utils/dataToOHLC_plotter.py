""" dataToOHLC_plotter.py 数据预处理器 """

import mplfinance as mpf
import datetime as dt

from pandas import DataFrame
from goInvest.utils import dataSource_picker as dsp
from typing import Literal


def plot_stock_ohlc(
    stock_code: str, period: Literal["daily", "weekly"], product_type: Literal["stock"]
):
    """绘制K线图的数据处理函数"""

    # 检查stock_code是一个六个字符的字符串
    if not (isinstance(stock_code, str) and len(stock_code) == 6):
        raise TypeError(f"{stock_code}必须是一个表示六位数的字符串")

    # 获取数据
    df_stock_dict = dsp.dataPicker.product_source_picker(
        stock_code, today_date=dt.date.today(), product_type=product_type
    )
    if all(value.empty for value in df_stock_dict.values()):
        raise ValueError(f"无法获取{stock_code}的K线数据，请检查股票代码是否正确，稍后重试")

    # 声明DataFrame对象
    ohlc_data = None
    # 从字典中取出取出数据
    ohlc_data = df_stock_dict[period][["日期", "开盘", "最高", "最低", "收盘", "成交量"]]
    # 调用数据预处理函数
    ohlc_data = ohlc_data_preprocess(ohlc_data)
    print(f"日K线图：预处理完成！")

    # 设置图表样式
    mc = mpf.make_marketcolors(
        up="r",
        down="g",
        wick="inherit",
        volume="inherit",
        edge="inherit",
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        gridcolor="black",
    )

    # 根据ohlc_data绘制出图像，保存到fig对象中
    ohlc_fig = None

    # 使用plot函数进行赋值时，必须使用returnfig=True参数
    if ohlc_data is not None:
        ohlc_fig = mpf.plot(
            ohlc_data,
            type="candle",
            style=style,
            fontscale=0.6,
            datetime_format="%Y-%m-%d",
            returnfig=True,
            volume=True,
            title=f"{stock_code}_{period}",
            xrotation=0,
            warn_too_much_data=2000,
        )
    print(f"{stock_code}{period[0].upper()}：绘制完成！")

    # 返回fig对象
    return ohlc_fig


def ohlc_data_preprocess(ohlc: DataFrame) -> DataFrame:
    """处理数据，使其符合mplfinance的要求"""

    # 转换列名
    ohlc = ohlc.rename(
        columns={
            "日期": "Date",
            "开盘": "Open",
            "最高": "High",
            "最低": "Low",
            "收盘": "Close",
            "成交量": "Volume",
        }
    )
    # 设置索引
    ohlc.set_index("Date", inplace=True)
    # 返回数据
    return ohlc
