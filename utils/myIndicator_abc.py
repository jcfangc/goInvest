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
        self.product_df_list = dp.dataPicker.product_source_picker(
            product_code=product_code, today_date=today_date, product_type=product_type
        )

    @abstractmethod
    def _remove_redundant_files(self) -> None:
        """删除多余的文件"""
        for period_short in ["D", "W"]:
            # 删除过往重复数据
            for file_name in os.listdir(self.data_path):
                # 防止stock_code和k_period_short为None时参与比较
                if self.product_code is not None:
                    # 指定K线过去的数据会被删除
                    if (
                        self.product_code
                        and period_short
                        and f"{self.indicator_name.value}"
                    ) in file_name:
                        # 取得文件绝对路径
                        absfile_path = os.path.join(self.data_path, file_name)
                        print(f"删除冗余文件\n>>>>{file_name}")
                        # os.remove只能处理绝对路径
                        os.remove(absfile_path)

    @abstractmethod
    def calculate_indicator(self) -> dict[str, DataFrame]:
        pass
