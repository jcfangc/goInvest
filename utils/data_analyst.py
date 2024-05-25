"""调用indicator文件夹中的技术指标函数，综合分析，给出买入卖出建议，输出文本、图像到data文件夹中"""

import datetime as dt
import pandas as pd
import json

from config import __BASE_PATH__
from utils import dataSource_picker as dp
from utils.enumeration_label import ProductType
from indicator import myBoll, myEMA, mySMA, mySRLine, myRSI

# 通用参数
params = {}


class StockAnalyst:
    """
    分析后返回上涨下跌预期
    """

    def __init__(self, stock_code: str, today_date: dt.date) -> None:
        self.product_code = stock_code
        self.today_date = today_date
        self.product_type = ProductType.Stock
        self.product_df_dict = dp.dataPicker.product_source_picker(
            product_code=self.product_code,
            today_date=self.today_date,
            product_type=self.product_type,
        )
        params["today_date"] = self.today_date
        params["product_code"] = self.product_code
        params["product_type"] = self.product_type
        params["product_df_dict"] = self.product_df_dict

    def analyze(self):
        default_analyze_list = ["SMA", "EMA", "Boll", "SRLine", "RSI"]
        analyze_list = self.analyze_list_from_config() or default_analyze_list
        if "SMA" in analyze_list:
            # 移动平均线分析
            self.result_list = mySMA.MySMA(data_path=None, **params).analyze()
        if "EMA" in analyze_list:
            # 指数移动平均线分析
            self.result_list.extend(myEMA.MyEMA(data_path=None, **params).analyze())
        if "Boll" in analyze_list:
            # 布林带分析
            self.result_list.extend(myBoll.MyBoll(data_path=None, **params).analyze())
        if "SRLine" in analyze_list:
            # 支撑线阻力线分析
            self.result_list.extend(
                mySRLine.MySRLine(data_path=None, **params).analyze()
            )
        if "RSI" in analyze_list:
            # RSI分析
            self.result_list.extend(myRSI.MyRSI(data_path=None, **params).analyze())

        print("开始计算综合分析结果...")
        # 创建一个空的DataFrame，用于存放综合的买入卖出建议
        df_comperhensive_judge = pd.DataFrame(
            index=self.result_list[0].index, columns=["daily", "weekly"]
        )

        # 初始化df_comperhensive_judge的值为0
        df_comperhensive_judge["daily"] = 0
        df_comperhensive_judge["weekly"] = 0

        # 遍历self.result_list
        for df in self.result_list:
            # print(df.head(30))
            # 将df中的daily列和weekly列的值加到df_comperhensive_judge中
            df_comperhensive_judge["daily"] += df["daily"]
            df_comperhensive_judge["weekly"] += df["weekly"]

        # 检查df_comperhensive_judge中的值是否存在nan
        if df_comperhensive_judge.isnull().values.any():
            # print(df_comperhensive_judge.head(50))
            raise ValueError("df_comperhensive_judge中存在nan值！")

        # 将df_comperhensive_judge中的值除以self.result_list的长度，得到平均值
        length = len(self.result_list)
        if length != 0:
            df_comperhensive_judge["daily"] = (
                df_comperhensive_judge["daily"] / length
            ).round(3)
            df_comperhensive_judge["weekly"] = (
                df_comperhensive_judge["weekly"] / length
            ).round(3)

        # 输出综合的买入卖出建议为csv文件，在strategy文件夹中
        with open(
            file=f"{__BASE_PATH__}\\data\\{self.product_type.value}\\{self.product_code}\\strategy\\{self.product_code}_comprehensive_anlysis.csv",
            mode="w",
            encoding="utf-8",
        ) as f:
            df_comperhensive_judge.to_csv(f)
            print(f"查看{self.product_code}的综合分析结果\n>>>>{f.name}")

    def analyze_list_from_config(self):
        """从配置文件获取分析列表"""
        with open(
            file=f"{__BASE_PATH__}\\config.json",
            mode="r",
            encoding="utf-8",
        ) as f:
            # 读取配置文件
            config_dict = json.load(f)
            # 检查是否存在analyze_list键
            if "analyze_list" not in config_dict.keys():
                # 如果不存在，返回默认值
                return None
            else:
                analyze_list = config_dict["analyze_list"]
                return analyze_list

    def analyze_list_to_config(self, analyze_list: list):
        """将分析列表写入配置文件"""
        with open(
            file=f"{__BASE_PATH__}\\config.json",
            mode="r",
            encoding="utf-8",
        ) as f:
            # 读取配置文件
            config_dict = json.load(f)
            # 更新配置文件
            config_dict["analyze_list"] = analyze_list
