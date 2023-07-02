if __name__ == "__main__":
    from __init__ import goInvest_path
else:
    from . import goInvest_path


import datetime as dt
import os

from pandas import DataFrame
from utils import dataSource_picker as dp
from typing import Literal


def my_boll(
    product_code: str,
    today_date: dt.date,
    time_window: Literal[5, 10, 20, 50, 150],
    product_type: Literal["stock"],
) -> dict[str, DataFrame]:
    """
    本函数计算布林带，k线图和布林带上下界有交集的数据记录下来
    """
    print(f"正在计算{product_code}的布林带...")

    # 获取指定股票的DataFrame数据
    stock_df = dp.dataPicker.product_source_picker(
        product_code=product_code, today_date=today_date, product_type=product_type
    )

    # 文件路径
    data_path = f"{goInvest_path}\\data\\kline\\indicator"

    for period_short in ["D", "W", "M"]:
        # 删除过往重复数据
        for file_name in os.listdir(data_path):
            # 防止stock_code和k_period_short为None时参与比较
            if product_code is not None:
                # 指定K线过去的数据会被删除
                if (product_code and period_short and "Boll") in file_name:
                    # 取得文件绝对路径
                    absfile_path = os.path.join(data_path, file_name)
                    print(f"删除冗余文件\n>>>>{file_name}")
                    # os.remove只能处理绝对路径
                    os.remove(absfile_path)

    # 定义一个字典，用于存放返回的不同周期bollinger数据
    df_bollinger_dict = {
        "daily": DataFrame(),
        "weekly": DataFrame(),
    }

    for period in ["daily", "weekly"]:
        # 获取移动平均线中间SMA
        mid_sma_series = dp.dataPicker.indicator_source_picker(
            product_code=product_code,
            today_date=today_date,
            time_window=time_window,  # 这个参数改变中轴线的值
            indicator_name="SMA",
            product_type=product_type,
        )[period][
            f"{time_window}均线"
        ]  # Name: 10均线, dtype: float64, index.name: 日期

        # 数据对齐
        stock_close_series = stock_df[period][
            "收盘"
        ]  # Name: 收盘, dtype: float64, index.name: 日期
        # 计算标准差
        sigma = (
            stock_close_series.rolling(time_window).std() * 0.948
        )  # 根据实际情况挑选最适合的修正系数0.948

        # 根据实际情况挑选最适合的系数
        k = 2
        # 根据中轴线，求出轨线
        # 上轨线
        up_track = mid_sma_series + k * sigma
        # 中轴线
        mid_axis = mid_sma_series
        # 下轨线
        down_track = mid_sma_series - k * sigma

        # 数据汇合
        df = DataFrame(
            {
                # 日期作为索引
                # 布林带数据
                "上轨": up_track.round(3),
                f"中轴{time_window}": mid_axis.round(3),
                "下轨": down_track.round(3),
            }
        )

        # 输出字典为csv文件
        with open(
            file=f"{data_path}\\{product_code}{period[0].upper()}_{today_date.strftime('%m%d')}_Boll.csv",
            mode="w",
            encoding="utf-8",
        ) as f:
            df.to_csv(f, index=True, encoding="utf-8")

        match period:
            case "daily":
                df_bollinger_dict["daily"] = df
            case "weekly":
                df_bollinger_dict["weekly"] = df

    return df_bollinger_dict


if __name__ == "__main__":
    # 测试
    my_boll(
        product_code="002230",
        today_date=dt.date.today(),
        time_window=10,
        product_type="stock",
    )
