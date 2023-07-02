if __name__ == "__main__":
    from __init__ import goInvest_path
else:
    from . import goInvest_path

import datetime as dt
import os

from pandas import DataFrame
from utils import dataSource_picker as dp
from typing import Literal


def my_SMA(
    product_code: str,
    today_date: dt.date,
    product_type: Literal["stock"],
) -> dict[str, DataFrame]:
    """
    本函数根据时间窗口计算移动平均线
    """
    print(f"正在计算{product_code}移动平均线...")

    # 文件路径
    data_path = f"{goInvest_path}\\data\\kline\\indicator"

    for period_short in ["D", "W", "M"]:
        # 删除过往重复数据
        for file_name in os.listdir(data_path):
            # 防止stock_code和k_period_short为None时参与比较
            if product_code is not None:
                # 指定K线过去的数据会被删除
                if (product_code and period_short and "SMA") in file_name:
                    # 取得文件绝对路径
                    absfile_path = os.path.join(data_path, file_name)
                    print(f"删除冗余文件\n>>>>{file_name}")
                    # os.remove只能处理绝对路径
                    os.remove(absfile_path)

    # 获取数据
    # 得到包含日K线/周K线的字典
    stock_df = dp.dataPicker.product_source_picker(
        product_code=product_code, today_date=today_date, product_type=product_type
    )

    # 定义一个字典，用于存放返回的不同周期sma数据
    df_sma_dict = {
        "daily": DataFrame(),
        "weekly": DataFrame(),
    }

    # 根据数据和时间窗口滚动计算SMA
    for period in ["daily", "weekly"]:
        # 5时间窗口，数值取三位小数
        sma_5 = stock_df[period]["收盘"].rolling(5).mean().round(3)
        # 10时间窗口，数值取三位小数
        sma_10 = stock_df[period]["收盘"].rolling(10).mean().round(3)
        # 20时间窗口，数值取三位小数
        sma_20 = stock_df[period]["收盘"].rolling(20).mean().round(3)
        # 50时间窗口，数值取三位小数
        sma_50 = stock_df[period]["收盘"].rolling(50).mean().round(3)
        # 150时间窗口，数值取三位小数
        sma_150 = stock_df[period]["收盘"].rolling(150).mean().round(3)

        # 均线数据汇合
        df_sma_dict[period] = DataFrame(
            {
                # 日期作为索引
                # 均线数据
                "5均线": sma_5,
                "10均线": sma_10,
                "20均线": sma_20,
                "50均线": sma_50,
                "150均线": sma_150,
            }
        )

        # 输出字典到csv文件
        with open(
            file=f"{data_path}\\{product_code}{period[0].upper()}_{today_date.strftime('%m%d')}_SMA.csv",
            mode="w",
            encoding="utf-8",
        ) as f:
            df_sma_dict[period].to_csv(f, index=True, encoding="utf-8")

    return df_sma_dict


if __name__ == "__main__":
    # 调用函数stock_SMA
    dict_sma_df = my_SMA(
        product_code="002230",
        today_date=dt.date.today(),
        product_type="stock",
    )
