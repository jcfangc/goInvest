"""调用indicator文件夹中的技术指标函数，综合分析，给出买入卖出建议，输出文本、图像到data文件夹中"""

import datetime as dt
import pandas as pd

from config import __BASE_PATH__
from utils import dataSource_picker as dp
from enumeration_label import ProductType
from indicator import myBoll, myEMA, mySMA


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

    def analyze(self):
        # 移动平均线分析
        self.result_list = mySMA.MySMA(
            data_path=None,
            today_date=self.today_date,
            product_code=self.product_code,
            product_type=self.product_type,
            product_df_dict=self.product_df_dict,
        ).analyze()
        # 指数移动平均线分析
        self.result_list.extend(
            myEMA.MyEMA(
                data_path=None,
                today_date=self.today_date,
                product_code=self.product_code,
                product_type=self.product_type,
                product_df_dict=self.product_df_dict,
            ).analyze()
        )
        # 布林带分析
        self.result_list.extend(
            myBoll.MyBoll(
                data_path=None,
                today_date=self.today_date,
                product_code=self.product_code,
                product_type=self.product_type,
                product_df_dict=self.product_df_dict,
            ).analyze()
        )

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
