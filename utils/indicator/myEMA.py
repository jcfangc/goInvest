if __name__ == "__main__":
    from __init__ import goInvest_path
else:
    from . import goInvest_path

import datetime as dt
import os

from pandas import DataFrame
from utils import dataSource_picker as dp
from typing import Literal


def my_EMA(
    product_code: str,
    today_date: dt.date,
    product_type: Literal["stock"],
) -> dict[str, DataFrame]:
    """
    本函数根据时间窗口计算移动平均线
    """
    print(f"正在计算{product_code}指数移动平均线...")

    # 文件路径
    data_path = f"{goInvest_path}\\data\\kline\\indicator"

    for period_short in ["D", "W"]:
        # 删除过往重复数据
        for file_name in os.listdir(data_path):
            # 防止stock_code和k_period_short为None时参与比较
            if product_code is not None:
                # 指定K线过去的数据会被删除
                if (product_code and period_short and "EMA") in file_name:
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
    df_ema_dict = {
        "daily": DataFrame(),
        "weekly": DataFrame(),
    }

    # 根据数据和时间窗口滚动计算SMA
    for period in ["daily", "weekly"]:
        closing_price = stock_df[period]["收盘"]
        # 5时间窗口，数值取三位小数
        ema_5 = closing_price.ewm(span=5, adjust=False).mean()
        # 10时间窗口，数值取三位小数
        ema_10 = closing_price.ewm(span=10, adjust=False).mean()
        # 20时间窗口，数值取三位小数
        ema_20 = closing_price.ewm(span=20, adjust=False).mean()
        # 50时间窗口，数值取三位小数
        ema_50 = closing_price.ewm(span=50, adjust=False).mean()
        # 150时间窗口，数值取三位小数
        ema_150 = closing_price.ewm(span=150, adjust=False).mean()

        # 均线数据汇合
        df_ema_dict[period] = DataFrame(
            {
                # 日期作为索引
                # 均线数据
                "5均线": ema_5.round(3),
                "10均线": ema_10.round(3),
                "20均线": ema_20.round(3),
                "50均线": ema_50.round(3),
                "150均线": ema_150.round(3),
            }
        )

        # 检查是否存在nan值
        if df_ema_dict[period].isnull().values.any():
            # 填充nan值
            df_ema_dict[period].fillna(value=0.0, inplace=True)

        # 输出字典到csv文件
        with open(
            file=f"{data_path}\\{product_code}{period[0].upper()}_{today_date.strftime('%m%d')}_EMA.csv",
            mode="w",
            encoding="utf-8",
        ) as f:
            df_ema_dict[period].to_csv(f, index=True, encoding="utf-8")

    return df_ema_dict


if __name__ == "__main__":
    # 调用函数stock_EMA
    dict_ema_df = my_EMA(
        product_code="002230",
        today_date=dt.date.today(),
        product_type="stock",
    )
