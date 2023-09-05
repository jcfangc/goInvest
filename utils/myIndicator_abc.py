"""myIndicator_abc.py 扮演一个接口，写一些指标中共性功能"""
if __name__ == "__main__":
    from __init__ import goInvest_path
else:
    from .indicator import goInvest_path

from abc import ABC, abstractmethod
from pandas import DataFrame
from utils import dataSource_picker as dp
from utils.enumeration_label import ProductType, IndicatorName

import datetime as dt
import os


class MyIndicator(ABC):
    def __init__(
        self,
        data_path: str | None,
        today_date: dt.date | None,
        product_code: str,
        product_type: ProductType,
        indicator_name: IndicatorName,
        product_df_dict: dict[str, DataFrame] | None,
    ) -> None:
        if data_path is None:
            # 默认路径
            self.data_path = f"{goInvest_path}\\data\\kline\\indicator"
        else:
            self.data_path = data_path
        if today_date is None:
            # 默认时间
            self.today_date = dt.date.today()
        else:
            self.today_date = today_date
        self.product_code = product_code
        self.product_type = product_type
        self.indicator_name = indicator_name
        if product_df_dict is None:
            self.product_df_dict = dp.dataPicker.product_source_picker(
                product_code=product_code,
                today_date=today_date,
                product_type=product_type,
            )
        else:
            self.product_df_dict = product_df_dict

    @abstractmethod
    def _remove_redundant_files(self) -> None:
        """
        删除多余的文件
        """
        for period_short in ["D", "W"]:
            # 删除过往重复数据
            for file_name in os.listdir(self.data_path):
                # 防止stock_code和k_period_short为None时参与比较
                if self.product_code is not None:
                    # 指定K线过去的数据会被删除
                    if (
                        self.product_code and period_short and self.indicator_name.value
                    ) in file_name and self.today_date.strftime(
                        "%m%d"
                    ) not in file_name:
                        # 取得文件绝对路径
                        absfile_path = os.path.join(self.data_path, file_name)
                        print(f"删除冗余文件\n>>>>{file_name}")
                        # os.remove只能处理绝对路径
                        os.remove(absfile_path)

    @abstractmethod
    def calculate_indicator(self) -> dict[str, DataFrame]:
        """
        重写本函数末尾可以调用save_indicator()函数，自动将计算好的指标数据保存到csv文件
        """
        pass

    @abstractmethod
    def analyze(self) -> list[DataFrame]:
        """
        本函数重写时，一开始记得调用pre_analyze()函数，获取指标数据\n
        本函数重写时，末尾可以调用save_strategy()函数，保存分析结果
        """
        pass

    def save_indicator(self, df_dict: dict[str, DataFrame]) -> None:
        """
        保存指标，可以在calculate_indicator()函数最后调用，自动将计算好的指标数据保存到csv文件
        """
        for period in df_dict.keys():
            # 检查是否存在nan值
            if df_dict[period].isnull().values.any():
                # 填充nan值
                df_dict[period].fillna(value=0.0, inplace=True)
            # 输出字典到csv文件
            with open(
                file=f"{self.data_path}\\{self.product_code}{period[0].upper()}_{self.today_date.strftime('%m%d')}_{self.indicator_name.value}.csv",
                mode="w",
                encoding="utf-8",
            ) as f:
                df_dict[period].to_csv(f, index=True, encoding="utf-8")

    def pre_analyze(self) -> dict[str, DataFrame]:
        """
        一些机械的重复性工作，在analyze()函数内部，先调用pre_analyze()函数，获取指标数据
        """
        return_dict = dp.dataPicker.indicator_source_picker(
            product_code=self.product_code,
            today_date=self.today_date,
            product_type=self.product_type,
            indicator_name=self.indicator_name,
            product_df_dict=self.product_df_dict,
        )

        if any(df.empty for df in return_dict.values()):
            raise ValueError("移动平均线数据为空！")

        return return_dict

    def save_strategy(self, df_judge: DataFrame, func_name: str) -> None:
        """
        输出df_sma_judge为csv文件，在strategy文件夹中\n
        在analyze()函数所调用的具体策略函数末尾，可以调用save_strategy()函数，保存分析结果
        """
        with open(
            f"{self.data_path}\\strategy\\{self.product_code}_{self.indicator_name.value}{func_name}_anlysis.csv",
            "w",
            encoding="utf-8",
        ) as f:
            df_judge.to_csv(f)
            print(
                f"查看{self.product_code}的'{self.indicator_name.value}{func_name}'分析结果\n>>>>{f.name}\n"
            )
