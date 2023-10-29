"""mySMA.py"""

import datetime as dt

from pandas import DataFrame
from utils.myIndicator_abc import MyIndicator
from utils.enumeration_label import ProductType, IndicatorName


class MySMA(MyIndicator):
    def __init__(
        self,
        data_path: str | None,
        today_date: dt.date | None,
        product_code: str,
        product_type: ProductType,
        product_df_dict: dict[str, DataFrame] | None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            today_date=today_date,
            product_code=product_code,
            product_type=product_type,
            indicator_name=IndicatorName.SMA,
            product_df_dict=product_df_dict,
        )

    def _remove_redundant_files(self) -> None:
        super()._remove_redundant_files()

    def calculate_indicator(self) -> dict[str, DataFrame]:
        """
        本函数根据时间窗口计算移动平均线
        """
        print(f"正在计算{self.product_code}移动平均线...")

        # 清理重复文件
        self._remove_redundant_files()

        # 定义一个字典，用于存放返回的不同周期sma数据
        df_sma_dict = {
            "daily": DataFrame(),
            "weekly": DataFrame(),
        }

        # 根据数据和时间窗口滚动计算SMA
        for period in ["daily", "weekly"]:
            closing_price = self.product_df_dict[period]["收盘"]
            # 5时间窗口，数值取三位小数
            sma_5 = closing_price.rolling(5).mean()
            # 10时间窗口，数值取三位小数
            sma_10 = closing_price.rolling(10).mean()
            # 20时间窗口，数值取三位小数
            sma_20 = closing_price.rolling(20).mean()
            # 50时间窗口，数值取三位小数
            sma_50 = closing_price.rolling(50).mean()
            # 150时间窗口，数值取三位小数
            sma_150 = closing_price.rolling(150).mean()

            # 均线数据汇合
            df_sma_dict[period] = DataFrame(
                {
                    # 日期作为索引
                    # 均线数据
                    "5均线": sma_5.round(3),
                    "10均线": sma_10.round(3),
                    "20均线": sma_20.round(3),
                    "50均线": sma_50.round(3),
                    "150均线": sma_150.round(3),
                }
            )

        # 保存均线数据
        super().save_indicator(df_dict=df_sma_dict)
        # 返回df_sma_dict
        return df_sma_dict

    def analyze(self) -> list[DataFrame]:
        # 获取股票K线SMA数据
        dict_sma = super().pre_analyze()
        # 调用策略函数
        sma_mutiline_judge = self._mutiline_strategy(dict_sma)
        # 返回策略结果
        return [sma_mutiline_judge]

    def _mutiline_strategy(
        self,
        dict_sma: dict[str, DataFrame],
    ) -> DataFrame:
        """
        多线策略，即多条均线同时作为看涨看空的依据\n
        在一段上涨态势中\n
        如果收盘价格跌破5均线，决策值设为-0.2\n
        如果收盘价格跌破10均线，决策值设为-0.3\n
        如果收盘价格跌破20均线，决策值设为-0.5\n
        如果收盘价格跌破50均线，决策值设为-1\n
        在一段下跌态势中\n
        如果收盘价格涨破5均线，决策值设为0.2\n
        如果收盘价格涨破10均线，决策值设为0.3\n
        如果收盘价格涨破20均线，决策值设为0.5\n
        如果收盘价格涨破50均线，决策值设为1\n
        """

        # 创建一个空的DataFrame
        # 第一列为“日期”索引，第二列（daily）第三列（weekly）为-1至1的SMA判断值
        df_sma_judge = DataFrame(
            index=dict_sma["daily"].index, columns=["daily", "weekly"]
        )

        # 初始化布林线判断值为0
        df_sma_judge["daily"] = 0
        df_sma_judge["weekly"] = 0

        # 获取股票数据
        ohlc_data = self.product_df_dict

        # 保存策略结果
        super().save_strategy(df_sma_judge, func_name=MySMA._mutiline_strategy.__name__)
        # 返回df_sma_judge
        return df_sma_judge


if __name__ == "__main__":
    MySMA(
        None, dt.date.today(), "002230", ProductType.Stock, None
    ).calculate_indicator()
