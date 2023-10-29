""" data_obtainer.py 数据获取器 """

import akshare as ak
import datetime as dt
import os
import pandas as pd

from pandas import DataFrame
from config import __BASE_PATH__


# 获取指定产品的日K/周K
def stock_data_obtainer(
    stock_code: str,
    today_date: dt.date,
) -> dict[str, DataFrame]:
    """
    输入指定股票代码和周期，返回周期K线数据（DataFrame）
    """
    # 最终要返回的所有K线数据，集合在字典中
    data_dict = {"daily": DataFrame(), "weekly": DataFrame()}

    # 遍历周期
    for k_period in ["daily", "weekly"]:
        # 获取指定股票的历史K线数据
        stock_df = ak.stock_zh_a_hist(
            # 股票代码
            symbol=stock_code,
            # K线周期
            period=k_period,
            # 复权
            adjust="qfq",
            # K线图起始日期
            start_date=(today_date - dt.timedelta(days=365 * 20)).strftime(
                "%Y%m%d"
            ),  # 20年前的今天
            # K线图结束日期
            end_date=today_date.strftime("%Y%m%d"),  # 今天
        )
        # 将日期列转换为datetime格式
        stock_df["日期"] = pd.to_datetime(stock_df["日期"])
        # 将日期列设置为索引
        stock_df.set_index("日期", inplace=True)

        # 数据路径
        data_path = f"{__BASE_PATH__}\\data\\stock\\{stock_code}\\kline"

        # 删除过往重复数据
        for file_name in os.listdir(data_path):
            # 防止stock_code和k_period_short为None时参与比较
            if stock_code is not None:
                # 指定K线过去的数据会被删除
                if (stock_code and k_period[0].upper()) in file_name:
                    # 取得文件绝对路径
                    absfile_path = os.path.join(data_path, file_name)
                    print(f"删除冗余文件\n>>>>{file_name}")
                    # os.remove只能处理绝对路径
                    os.remove(absfile_path)

        # 输出数据
        with open(
            f"{data_path}\\{stock_code}{k_period[0].upper()}_{today_date.strftime('%m%d')}.csv",
            "w",
            encoding="utf-8",
        ) as f:
            # 输出，index=False代表不输出dataFrame对象中的序号
            stock_df.to_csv(f, index=True, encoding="utf-8")
            # 返回输出位置
            print(f"K线数据更新\n>>>>{os.path.abspath(f.name)}")

        # 将数据存入字典
        match k_period:
            case "daily":
                data_dict["daily"] = stock_df
            case "weekly":
                data_dict["weekly"] = stock_df

    # 返回字典
    return data_dict


if __name__ == "__main__":
    # 测试
    stock_data_obtainer("002230", dt.date.today())
