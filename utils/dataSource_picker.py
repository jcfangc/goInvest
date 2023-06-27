""" dataSource_picker.py 本模块用于自动寻找相应股票的数据源 """

import datetime as dt
import os
import pandas as pd


from pandas import DataFrame
from goInvest.utils import data_obtainer as do
from goInvest.utils.indicator import mySMA
from goInvest.utils.indicator import myBoll
from typing import Literal


class dataPicker:
    """一个寻找指定数据源的类"""

    @staticmethod
    def product_source_picker(
        product_code: str,
        today_date: dt.date,
        product_type: Literal["stock"],
    ) -> dict[str, DataFrame]:
        """
        自动寻找相应产品的数据源
        """
        match product_type:
            case "stock":
                print("正在自动寻找'股票'数据源...")

        # 定义一个字典，用于存放返回的股票K线数据
        df_product_dict = {
            "daily": DataFrame(),
            "weekly": DataFrame(),
        }

        # 定义数据源文件夹路径
        data_path = "goInvest\\data\\kline"
        # 查询目标文件夹是否存在相关数据
        # 如果不存在相关数据，自行获取
        for file_name in os.listdir(data_path):
            for k_period_short in ["D", "W"]:
                if (
                    f"{product_code}{k_period_short}_{today_date.strftime('%m%d')}.csv"
                    == file_name
                ):
                    file_path = os.path.join(data_path, file_name)
                    match k_period_short:
                        case "D":
                            print(f"最新日K数据已经保存在{file_path}，正在读取...")
                            df_product_dict["daily"] = pd.read_csv(
                                file_path,
                            )
                        case "W":
                            print(f"最新周K数据已经保存在{file_path}，正在读取...")
                            df_product_dict["weekly"] = pd.read_csv(
                                file_path,
                            )

        # 检查字典内的数据是不是都是空的DataFrame，如果是，自行获取
        if all(value.empty for value in df_product_dict.values()):
            if product_type == "stock":
                df_product_dict = do.stock_data_obtainer(
                    stock_code=product_code,
                    today_date=today_date,
                )
        else:
            # 遍历字典中的值，将每个DataFrame的“日期”列转换为datetime格式
            for df_product in df_product_dict.values():
                # 将“日期”列转换为datetime格式
                df_product["日期"] = pd.to_datetime(df_product["日期"])
                # 将“日期”列设置为索引
                df_product.set_index("日期", inplace=True)

        # 返回数据字典
        return df_product_dict

    @staticmethod
    def indicator_source_picker(
        product_code: str,
        today_date: dt.date,
        time_window: int,
        product_type: Literal["stock"],
        indicator_name: Literal["SMA", "Boll"],
    ) -> dict[str, DataFrame]:
        """
        寻找相应技术指标数据源\n
        当indicator_name为"SMA"时，time_window为移动平均线的‘最小’时间窗口，程序会自动生成20，50，150周期的时间窗口\n
        当indicator_name为"Boll"时，time_window为布林线的‘中轴线’时间窗口\n
        """

        match indicator_name:
            case "SMA":
                print("正在自动寻找'移动平均线'数据源...")
            case "Boll":
                print("正在自动寻找'布林线'数据源...")

        # 定义一个字典，用于存放返回的sma数据
        dict_indicator = {
            "daily": DataFrame(),
            "weekly": DataFrame(),
        }

        # sma数据源文件夹路径
        indicator_path = "goInvest\\data\\kline\\indicator"

        for file_name in os.listdir(indicator_path):
            # 依次检查日K、周K的sma数据源是否存在
            for k_period_short in ["D", "W"]:
                if (
                    f"{product_code}{k_period_short}_{today_date.strftime('%m%d')}_{indicator_name}.csv"
                    == file_name
                ):
                    file_path = os.path.join(indicator_path, file_name)
                    match k_period_short:
                        case "D":
                            print(f"最新日{indicator_name}数据已经保存在{file_path}，正在读取...")
                            dict_indicator["daily"] = pd.read_csv(
                                file_path,
                            )
                        case "W":
                            print(f"最新周{indicator_name}数据已经保存在{file_path}，正在读取...")
                            dict_indicator["weekly"] = pd.read_csv(
                                file_path,
                            )

        # 检查字典内的数据是不是都是空的DataFrame，如果是，自行获取
        if all(value.empty for value in dict_indicator.values()):
            if indicator_name == "Boll":
                dict_indicator = myBoll.my_boll(
                    product_code=product_code,
                    today_date=today_date,
                    time_window=time_window,
                    product_type=product_type,
                )
            elif indicator_name == "SMA":
                dict_indicator = mySMA.my_SMA(
                    product_code=product_code,
                    today_date=today_date,
                    time_window=time_window,
                    product_type=product_type,
                )
        else:
            # 遍历字典中的值，将每个DataFrame的“日期”列转换为datetime格式
            for df_indicator in dict_indicator.values():
                # 将“日期”列转换为datetime格式
                df_indicator["日期"] = pd.to_datetime(df_indicator["日期"])
                # 将“日期”列设置为索引
                df_indicator.set_index("日期", inplace=True)

        # 返回数据字典
        return dict_indicator
