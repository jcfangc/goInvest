""" dataSource_picker.py 本模块用于自动寻找相应股票的数据源 """

import datetime as dt
import os
import pandas as pd

from pandas import DataFrame
from utils import data_obtainer as do
from utils.enumeration_label import ProductType, IndicatorName, StrategyName
from config import __BASE_PATH__


class dataPicker:
    """一个寻找指定数据源的类"""

    def __init__(self) -> None:
        pass

    @staticmethod
    def product_source_picker(
        product_code: str,
        today_date: dt.date | None,
        product_type: ProductType,
    ) -> dict[str, DataFrame]:
        """
        自动寻找相应产品的数据源
        """
        match product_type:
            case ProductType.Stock:
                print("正在自动寻找'股票'数据源...")

        today_date = dt.date.today() if today_date is None else today_date

        # 定义一个字典，用于存放返回的股票K线数据
        df_product_dict = {
            "daily": DataFrame(),
            "weekly": DataFrame(),
        }

        # 定义数据源文件夹路径
        data_path = (
            f"{__BASE_PATH__}\\data\\{product_type.value}\\{product_code}\\kline"
        )
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
        if any(value.empty for value in df_product_dict.values()):
            print(f"发现最新数据缺失，重新获取'{product_code}'所有K线数据...\n")
            if product_type == ProductType.Stock:
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
        product_type: ProductType,
        indicator_name: IndicatorName,
        product_df_dict: dict[str, DataFrame] | None,
    ) -> dict[str, DataFrame]:
        """
        寻找相应技术指标数据源\n
        """

        # 通用参数
        params = {}
        params["product_code"] = product_code
        params["today_date"] = today_date
        params["product_type"] = product_type
        params["product_df_dict"] = product_df_dict

        match indicator_name:
            case IndicatorName.EMA:
                print("正在自动寻找'指数移动平均线'数据源...")
            case IndicatorName.SMA:
                print("正在自动寻找'移动平均线'数据源...")
            case IndicatorName.Boll:
                print("正在自动寻找'布林线'数据源...")
            case IndicatorName.RSI:
                print("正在自动寻找'相对强弱指标'数据源...")
            case IndicatorName.SRLine:
                print("正在自动寻找'支撑/阻力线'数据源...")

        # 定义一个字典，用于存放返回的数据
        dict_indicator = {
            "daily": DataFrame(),
            "weekly": DataFrame(),
        }

        # 数据源文件夹路径
        indicator_path = (
            f"{__BASE_PATH__}\\data\\{product_type.value}\\{product_code}\\indicator"
        )
        for file_name in os.listdir(indicator_path):
            # 依次检查日K、周K的数据源是否存在
            for k_period_short in ["D", "W"]:
                if (
                    f"{product_code}{k_period_short}_{today_date.strftime('%Y%m%d')}_{indicator_name.value}.csv"
                    == file_name
                ):
                    file_path = os.path.join(indicator_path, file_name)
                    match k_period_short:
                        case "D":
                            print(
                                f"最新日'{indicator_name.value}'数据已经保存在{file_path}，正在读取..."
                            )
                            dict_indicator["daily"] = pd.read_csv(
                                file_path,
                            )
                        case "W":
                            print(
                                f"最新周'{indicator_name.value}'数据已经保存在{file_path}，正在读取..."
                            )
                            dict_indicator["weekly"] = pd.read_csv(
                                file_path,
                            )

        # 检查字典内的数据是不是都是空的DataFrame，如果是，自行获取
        if all(value.empty for value in dict_indicator.values()):
            if indicator_name == IndicatorName.Boll:
                from indicator import myBoll

                dict_indicator = myBoll.MyBoll(None, **params).calculate_indicator()
            elif indicator_name == IndicatorName.SMA:
                from indicator import mySMA

                dict_indicator = mySMA.MySMA(None, **params).calculate_indicator()
            elif indicator_name == IndicatorName.EMA:
                from indicator import myEMA

                dict_indicator = myEMA.MyEMA(None, **params).calculate_indicator()
            elif indicator_name == IndicatorName.RSI:
                from indicator import myRSI

                dict_indicator = myRSI.MyRSI(None, **params).calculate_indicator()
            elif indicator_name == IndicatorName.SRLine:
                from indicator import mySRLine

                dict_indicator = mySRLine.MySRLine(None, **params).calculate_indicator()
        else:
            # 遍历字典中的值，将每个DataFrame的“日期”列转换为datetime格式
            for df_indicator in dict_indicator.values():
                # 将“日期”列转换为datetime格式
                df_indicator["日期"] = pd.to_datetime(df_indicator["日期"])
                # 将“日期”列设置为索引
                df_indicator.set_index("日期", inplace=True)

        # 返回数据字典
        return dict_indicator

    @staticmethod
    def strategy_source_picker(
        product_code: str,
        today_date: dt.date,
        product_type: ProductType,
        strategy_name: StrategyName,
        product_df_dict: dict[str, DataFrame] | None,
    ) -> DataFrame:
        """
        寻找相应策略数据源\n
        """

        # 通用参数
        params = {}
        params["product_code"] = product_code
        params["today_date"] = today_date
        params["product_type"] = product_type
        params["product_df_dict"] = product_df_dict

        match strategy_name:
            case StrategyName.Boll_BLL:
                print("正在自动寻找'布林线策略'数据源...")
            case StrategyName.SMA_MTL:
                print("正在自动寻找'移动平均线多均线策略'数据源...")
            case StrategyName.EMA_MTL:
                print("正在自动寻找'指数移动平均线多均线策略'数据源...")
            case StrategyName.SRLine_PSS:
                print("正在自动寻找'支撑/阻力线压力区策略'数据源...")

        # 定义一个字典，用于存放返回的数据
        df_strategy = DataFrame()

        # 数据源文件夹路径
        strategy_path = (
            f"{__BASE_PATH__}\\data\\{product_type.value}\\{product_code}\\strategy"
        )
        for file_name in os.listdir(strategy_path):
            if strategy_name.value in file_name:
                file_path = os.path.join(strategy_path, file_name)
                print(f"'{strategy_name.value[1:]}'数据已经保存在{file_path}，正在读取...")
                df_strategy = pd.read_csv(
                    file_path,
                )

        # 检查字典内的数据是不是都是空的DataFrame，如果是，自行获取
        if df_strategy.empty:
            if strategy_name == StrategyName.Boll_BLL.value:
                from indicator import myBoll

                myBoll.MyBoll(None, **params).analyze()
            elif strategy_name == StrategyName.SMA_MTL.value:
                from indicator import mySMA

                mySMA.MySMA(None, **params).analyze()
            elif strategy_name == StrategyName.EMA_MTL.value:
                from indicator import myEMA

                myEMA.MyEMA(None, **params).analyze()
            elif strategy_name == StrategyName.SRLine_PSS.value:
                from indicator import mySRLine

                mySRLine.MySRLine(None, **params).analyze()

            # 递归
            df_strategy = dataPicker.strategy_source_picker(
                strategy_name=strategy_name, **params
            )

        else:
            # 将“日期”列转换为datetime格式
            df_strategy["日期"] = pd.to_datetime(df_strategy["日期"])
            # 将“日期”列设置为索引
            df_strategy.set_index("日期", inplace=True)

        # 返回数据字典
        return df_strategy
