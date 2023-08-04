if __name__ == "__main__":
    from __init__ import goInvest_path
else:
    from . import goInvest_path

import datetime as dt
import os

from pandas import DataFrame, Series
from utils import dataSource_picker as dp
from typing import Literal


def _wilder_smoothing_method(values: Series, period: int) -> Series:
    """Wilder's Smoothing Value (WSV) = [(previous WSV) * (n - 1) + current value] / n"""
    # 创建一个新的Series，用于存储平滑后的值
    smoothed_values = values.copy()
    # 第一个有效值使用sma计算
    first_wsv = values[:period].mean()
    # 覆盖第一个值
    smoothed_values[0] = first_wsv
    # 从第二个值开始，使用平滑算法计算RSI
    for i in range(1, len(values)):
        wsv = (smoothed_values[i - 1] * (period - 1) + smoothed_values[i]) / period
        smoothed_values[i] = wsv
    # 返回平滑后的值
    return smoothed_values


def my_rsi(
    product_code: str,
    today_date: dt.date,
    product_type: Literal["stock"],
) -> dict[str, DataFrame]:
    """
    本函数计算相对强弱指标，返回一个字典，包含不同周期的相对强弱指标数据
    """
    print(f"正在计算{product_code}的相对强弱指标...")

    # 定义短期和长期的时间周期
    short_term = 7
    long_term = 14

    # 获取指定股票的DataFrame数据
    dict_stock_df = dp.dataPicker.product_source_picker(
        product_code=product_code, today_date=today_date, product_type=product_type
    )

    # 文件路径
    data_path = f"{goInvest_path}\\data\\kline\\indicator"

    for period_short in ["D", "W"]:
        # 删除过往重复数据
        for file_name in os.listdir(data_path):
            # 防止stock_code和k_period_short为None时参与比较
            if product_code is not None:
                # 指定K线过去的数据会被删除
                if (product_code and period_short and "RSI") in file_name:
                    # 取得文件绝对路径
                    absfile_path = os.path.join(data_path, file_name)
                    print(f"删除冗余文件\n>>>>{file_name}")
                    # os.remove只能处理绝对路径
                    os.remove(absfile_path)

    # 定义一个字典，用于存放返回的不同周期rsi数据
    dict_rsi_df = {}

    # 遍历不同的时间周期，计算相对强弱指标
    for period in dict_stock_df.keys():
        # 获取daily和weekly的ohlc数据
        df_ohlc = dict_stock_df[period]
        # 创建一个新的DataFrame，用于存储RSI_short_term和RSI_long_term的数据
        df_rsi = DataFrame(
            index=dict_stock_df[period].index,
            columns=[
                "difference",
                f"avg_gain_{short_term}",
                f"avg_loss_{short_term}",
                f"avg_gain_{long_term}",
                f"avg_loss_{long_term}",
                f"RSI_{short_term}",
                f"RSI_{long_term}",
                f"SMA_{short_term}",
            ],
        )

        # 滚动计算相邻两天的收盘价之差，存储在difference列中，注意，第一行的difference为NaN
        df_rsi["difference"] = df_ohlc["收盘"].diff().round(4)

        # 将上涨的差价和下跌的差价分别存储在positive_difference和negative_difference中
        positive_difference = df_rsi["difference"].apply(lambda x: x if x > 0 else 0)
        negative_difference = df_rsi["difference"].apply(
            lambda x: abs(x) if x < 0 else 0
        )
        # 计算short_term和long_term的平均涨幅和平均跌幅
        df_rsi[f"avg_gain_{short_term}"] = _wilder_smoothing_method(
            positive_difference, short_term
        ).round(4)
        df_rsi[f"avg_loss_{short_term}"] = _wilder_smoothing_method(
            negative_difference, short_term
        ).round(4)
        df_rsi[f"avg_gain_{long_term}"] = _wilder_smoothing_method(
            positive_difference, long_term
        ).round(4)
        df_rsi[f"avg_loss_{long_term}"] = _wilder_smoothing_method(
            negative_difference, long_term
        ).round(4)
        # 计算RSI_short_term和RSI_long_term
        df_rsi[f"RSI_{short_term}"] = (
            df_rsi[f"avg_gain_{short_term}"]
            / (df_rsi[f"avg_gain_{short_term}"] + df_rsi[f"avg_loss_{short_term}"])
            * 100
        ) - (
            0.03 if short_term == 7 else 0
        )  # 根据实际情况，将RSI_short_term的值适当调整
        df_rsi[f"RSI_{short_term}"] = df_rsi[f"RSI_{short_term}"].round(3)

        df_rsi[f"RSI_{long_term}"] = (
            df_rsi[f"avg_gain_{long_term}"]
            / (df_rsi[f"avg_gain_{long_term}"] + df_rsi[f"avg_loss_{long_term}"])
            * 100
        ) - (
            0.055 if long_term == 14 else 0
        )  # 根据实际情况，将RSI_long_term的值适当调整
        df_rsi[f"RSI_{long_term}"] = df_rsi[f"RSI_{long_term}"].round(3)

        # 计算SMA_short_term
        df_rsi[f"SMA_{short_term}"] = (
            df_rsi[f"RSI_{short_term}"].rolling(short_term).mean()
        ).round(3)

        # 检查是否存在NaN值
        if df_rsi.isnull().values.any():
            # 填充为0.0
            df_rsi.fillna(0.0, inplace=True)

        # 将df_rsi添加到dict_rsi_df中
        dict_rsi_df[period] = df_rsi

        # 将df_rsi保存为csv文件
        with open(
            file=f"{goInvest_path}\\data\\kline\\indicator\\{product_code}{period[0].upper()}_{today_date.strftime('%m%d')}_RSI.csv",
            mode="w",
            encoding="utf-8",
        ) as f:
            df_rsi.to_csv(f)
            # print(len(quque_short_term))
            # print(len(quque_long_term))

    # 返回dict_rsi_df
    return dict_rsi_df


if __name__ == "__main__":
    my_rsi(
        product_code="002230",
        today_date=dt.datetime.today().date(),
        product_type="stock",
    )
