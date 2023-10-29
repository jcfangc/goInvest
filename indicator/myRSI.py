"""myRSI.py"""

import datetime as dt

from pandas import DataFrame, Series
from utils.myIndicator_abc import MyIndicator
from utils.enumeration_label import ProductType, IndicatorName


class MyRSI(MyIndicator):
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
            indicator_name=IndicatorName.RSI,
            product_df_dict=product_df_dict,
        )

    def _remove_redundant_files(self) -> None:
        super()._remove_redundant_files()

    def calculate_indicator(self) -> dict[str, DataFrame]:
        """
        本函数计算相对强弱指标，返回一个字典，包含不同周期的相对强弱指标数据
        """
        print(f"正在计算{self.product_code}的相对强弱指标...")

        # 清理重复文件
        self._remove_redundant_files()

        # 定义短期和长期的时间周期
        self.short_term = 7
        self.long_term = 14

        # 定义一个字典，用于存放返回的不同周期rsi数据
        dict_rsi_df = {}

        # 遍历不同的时间周期，计算相对强弱指标
        for period in self.product_df_dict.keys():
            # 获取daily和weekly的ohlc数据
            df_ohlc = self.product_df_dict[period]
            # 创建一个新的DataFrame，用于存储RSI_short_term和RSI_long_term的数据
            df_rsi = DataFrame(
                index=self.product_df_dict[period].index,
                columns=[
                    "difference",
                    f"avg_gain_{self.short_term}",
                    f"avg_loss_{self.short_term}",
                    f"avg_gain_{self.long_term}",
                    f"avg_loss_{self.long_term}",
                    f"RSI_{self.short_term}",
                    f"RSI_{self.long_term}",
                    f"SMA_{self.short_term}",
                ],
            )

            # 滚动计算相邻两天的收盘价之差，存储在difference列中，注意，第一行的difference为NaN
            df_rsi["difference"] = df_ohlc["收盘"].diff().round(4)

            # 将上涨的差价和下跌的差价分别存储在positive_difference和negative_difference中
            positive_difference = df_rsi["difference"].apply(
                lambda x: x if x > 0 else 0
            )
            negative_difference = df_rsi["difference"].apply(
                lambda x: abs(x) if x < 0 else 0
            )
            # 计算short_term和long_term的平均涨幅和平均跌幅
            df_rsi[f"avg_gain_{self.short_term}"] = self._wilder_smoothing_method(
                positive_difference, self.short_term
            ).round(4)
            df_rsi[f"avg_loss_{self.short_term}"] = self._wilder_smoothing_method(
                negative_difference, self.short_term
            ).round(4)
            df_rsi[f"avg_gain_{self.long_term}"] = self._wilder_smoothing_method(
                positive_difference, self.long_term
            ).round(4)
            df_rsi[f"avg_loss_{self.long_term}"] = self._wilder_smoothing_method(
                negative_difference, self.long_term
            ).round(4)
            # 计算RSI_short_term和RSI_long_term
            df_rsi[f"RSI_{self.short_term}"] = (
                df_rsi[f"avg_gain_{self.short_term}"]
                / (
                    df_rsi[f"avg_gain_{self.short_term}"]
                    + df_rsi[f"avg_loss_{self.short_term}"]
                )
                * 100
            ) - (
                0.03 if self.short_term == 7 else 0
            )  # 根据实际情况，将RSI_short_term的值适当调整
            df_rsi[f"RSI_{self.short_term}"] = df_rsi[f"RSI_{self.short_term}"].round(3)

            df_rsi[f"RSI_{self.long_term}"] = (
                df_rsi[f"avg_gain_{self.long_term}"]
                / (
                    df_rsi[f"avg_gain_{self.long_term}"]
                    + df_rsi[f"avg_loss_{self.long_term}"]
                )
                * 100
            ) - (
                0.055 if self.long_term == 14 else 0
            )  # 根据实际情况，将RSI_long_term的值适当调整
            df_rsi[f"RSI_{self.long_term}"] = df_rsi[f"RSI_{self.long_term}"].round(3)

            # 计算SMA_short_term
            df_rsi[f"SMA_{self.short_term}"] = (
                df_rsi[f"RSI_{self.short_term}"].rolling(self.short_term).mean()
            ).round(3)

            # 检查是否存在NaN值
            if df_rsi.isnull().values.any():
                # 填充为0.0
                df_rsi.fillna(0.0, inplace=True)

            # 将df_rsi添加到dict_rsi_df中
            dict_rsi_df[period] = df_rsi

        # 保存指标
        super().save_indicator(df_dict=dict_rsi_df)
        # 返回dict_rsi_df
        return dict_rsi_df

    def _wilder_smoothing_method(self, values: Series, period: int) -> Series:
        """Wilder's Smoothing Value (WSV) = [(previous WSV) * (n - 1) + current value] / n"""
        # 创建一个新的Series，用于存储平滑后的值
        smoothed_values = values.copy()
        # 第一个有效值使用sma计算
        first_wsv = values[:period].mean()
        # 覆盖第一个值
        smoothed_values[0] = first_wsv
        # 从第二个值开始，使用平滑算法计算RSI
        for i in range(1, len(values)):
            wsv = (smoothed_values[i - 1] * (period - 1) + smoothed_values[i]) / period
            smoothed_values[i] = wsv
        # 返回平滑后的值
        return smoothed_values

    def analyze(self) -> list[DataFrame]:
        # 获取股票K线RSI数据
        dict_rsi_data = super().pre_analyze()
        # 调用策略函数
        rsi_obs_strategy = self._over_bought_sold_strategy(dict_rsi_data)
        # 返回策略
        return [rsi_obs_strategy]

    def _over_bought_sold_strategy(
        self, dict_rsi_data: dict[str, DataFrame]
    ) -> DataFrame:
        """
        超买：RSI指标下穿70，上涨预期1\n
        超卖：RSI指标上穿30，下跌预期-1\n
        """
        # 创建一个空的DataFrame，用于存放买入卖出数值
        df_rsi_judge = DataFrame(
            index=dict_rsi_data["daily"].index, columns=["daily", "weekly"]
        )
        # 初始化买入卖出数值为0.0
        df_rsi_judge["daily"] = 0.0
        df_rsi_judge["weekly"] = 0.0

        # 保存策略
        super().save_strategy(
            df_judge=df_rsi_judge, func_name=MyRSI._over_bought_sold_strategy.__name__
        )
        # 返回df_rsi_judge
        return df_rsi_judge


if __name__ == "__main__":
    MyRSI(
        None, dt.date.today(), "002230", ProductType.Stock, None
    ).calculate_indicator()
