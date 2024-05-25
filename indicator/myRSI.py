"""myRSI.py"""

if __name__ == "__main__":
    import sys
    import os

    # 将上级目录加入sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

import datetime as dt

from pandas import DataFrame, Series
from utils.myIndicator_abc import MyIndicator
from utils.enumeration_label import ProductType, IndicatorName
from typing import Optional
from utils.data_functionalizer import DataFunctionalizer as dfunc

# 本指标的参数
params = {}


class MyRSI(MyIndicator):
    def __init__(
        self,
        data_path: Optional[str],
        today_date: Optional[dt.date],
        product_code: str,
        product_type: ProductType,
        product_df_dict: Optional[dict[str, DataFrame]],
    ) -> None:
        params["today_date"] = today_date
        params["product_code"] = product_code
        params["product_type"] = product_type
        params["product_df_dict"] = product_df_dict
        super().__init__(
            data_path=data_path, indicator_name=IndicatorName.RSI, **params
        )

        # 本指标的参数
        default_indicator_config_value = {"short_term": 7, "long_term": 14}
        self.indicator_config_value = (
            super().read_from_config(None) or default_indicator_config_value
        )

        # 定义短期和长期的时间周期
        self.short_term = self.indicator_config_value["short_term"]
        self.long_term = self.indicator_config_value["long_term"]

    def _remove_redundant_files(self) -> None:
        super()._remove_redundant_files()

    def calculate_indicator(self) -> dict[str, DataFrame]:
        """
        本函数计算相对强弱指标，返回一个字典，包含不同周期的相对强弱指标数据
        """
        print(f"正在计算{self.product_code}的相对强弱指标...")

        # 清理重复文件
        self._remove_redundant_files()

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
            for term in [self.short_term, self.long_term]:
                df_rsi[f"RSI_{term}"] = (
                    df_rsi[f"avg_gain_{term}"]
                    / (df_rsi[f"avg_gain_{term}"] + df_rsi[f"avg_loss_{term}"])
                    * 100
                )
                df_rsi[f"RSI_{term}"] = df_rsi[f"RSI_{term}"].round(3)

            # 计算SMA_short_term
            df_rsi[f"SMA_{self.short_term}"] = (
                df_rsi[f"RSI_{self.short_term}"].rolling(self.short_term).mean()
            ).round(3)

            # 删除辅助计算的列
            df_rsi.drop(
                columns=[
                    "difference",
                    f"avg_gain_{self.short_term}",
                    f"avg_loss_{self.short_term}",
                    f"avg_gain_{self.long_term}",
                    f"avg_loss_{self.long_term}",
                    f"RSI_{self.short_term}",
                ],
                inplace=True,
            )

            # 检查是否存在NaN值
            if df_rsi.isnull().values.any():
                # 填充为0.0
                df_rsi.fillna(0.0, inplace=True)

            # 将df_rsi添加到dict_rsi_df中
            dict_rsi_df[period] = df_rsi

        # 保存指标
        super().save_indicator(
            df_dict=dict_rsi_df, indicator_value_config_dict=self.indicator_config_value
        )
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
        dict_rsi_data = super().get_dict()
        # 调用策略函数
        rsi_et_strategy = self._exceedingly_trade_strategy(dict_rsi_data)
        rsi_vsma_strategy = self._versus_sma_strategy(dict_rsi_data)
        # 返回策略
        return [rsi_et_strategy, rsi_vsma_strategy]

    # 策略函数名请轻易不要修改！！！若修改，需要同时修改枚举类内的StrategyName！！！
    def _exceedingly_trade_strategy(
        self, dict_rsi_data: dict[str, DataFrame]
    ) -> DataFrame:
        """
        超买：RSI指标长期（默认14）下穿阈值1（默认70），预期-1\n
        超卖：RSI指标长期（默认14）上穿阈值2（默认30），预期1\n
        """
        # 策略参数
        default_strategy_config_value = {"threshold_1": 75.0, "threshold_2": 25.0}
        strategy_config_value = (
            super().read_from_config(MyRSI._exceedingly_trade_strategy.__name__)
            or default_strategy_config_value
        )

        # 创建一个空的DataFrame，用于涨跌预期判断值存放
        df_rsi_judge = DataFrame(
            index=dict_rsi_data["daily"].index, columns=["daily", "weekly"]
        )
        # 初始化买入卖出数值为0.0
        df_rsi_judge["daily"] = 0.0
        df_rsi_judge["weekly"] = 0.0

        # 获取阈值
        threshold_1 = strategy_config_value["threshold_1"]
        threshold_2 = strategy_config_value["threshold_2"]

        # 遍历daily和weekly的RSI数据
        for period in dict_rsi_data.keys():
            # 创建Series，长度和dict_rsi_data["daily"]相同，值为阈值
            threshold_1_series = Series(threshold_1, index=dict_rsi_data[period].index)
            threshold_2_series = Series(threshold_2, index=dict_rsi_data[period].index)

            # 获取RSI数据
            df_rsi = dict_rsi_data[period]
            # 获取RSI_long_term的数据 Series
            rsi_long_term = df_rsi[f"RSI_{self.long_term}"]

            # 调用check_cross函数，判断RSI_long_term是否下穿阈值1
            df_threshold_1_cross = dfunc.check_cross(
                threshold_1_series, rsi_long_term, up_down=True
            )
            # 取下穿阈值1的日期
            down_threshold_1_date = df_threshold_1_cross["下穿"]
            # 调用check_cross函数，判断RSI_long_term是否上穿阈值2
            df_threshold_2_cross = dfunc.check_cross(
                threshold_2_series, rsi_long_term, up_down=True
            )
            # 取上穿阈值2的日期
            up_threshold_2_date = df_threshold_2_cross["上穿"]

            # 将对应下穿的期望值设为-1
            df_rsi_judge.loc[down_threshold_1_date, period] = -1
            # 将对应上穿的期望值设为1
            df_rsi_judge.loc[up_threshold_2_date, period] = 1

        # 保存策略
        super().save_strategy(
            df_judge=df_rsi_judge,
            func_name=MyRSI._exceedingly_trade_strategy.__name__,
            strategy_config_value_dict=strategy_config_value,
            saperate=True,
        )

        # 返回df_rsi_judge
        return df_rsi_judge

    # 策略函数名请轻易不要修改！！！若修改，需要同时修改枚举类内的StrategyName！！！
    def _versus_sma_strategy(self, dict_rsi_data: dict[str, DataFrame]) -> DataFrame:
        """
        上穿短期（默认7）SMA，预期为正数\n
        下穿短期（默认7）SMA，预期为负数\n
        """

        # 创建一个空的DataFrame，用于涨跌预期判断值存放
        df_rsi_judge = DataFrame(
            index=dict_rsi_data["daily"].index, columns=["daily", "weekly"]
        )
        # 初始化买入卖出数值为0.0
        df_rsi_judge["daily"] = 0.0
        df_rsi_judge["weekly"] = 0.0

        # 遍历daily和weekly的RSI数据
        for period in dict_rsi_data.keys():
            # 获取RSI数据
            df_rsi = dict_rsi_data[period]
            # 获取RSI_long_term的数据 Series
            rsi_long_term = df_rsi[f"RSI_{self.long_term}"]
            # # 平滑处理
            # rsi_long_term = dfunc.smoother_series(rsi_long_term, 5)
            # 获取SMA_short_term的数据 Series
            sma_short_term = df_rsi[f"SMA_{self.short_term}"]
            # # 平滑处理
            # sma_short_term = dfunc.smoother_series(sma_short_term, 5)

            # 调用check_cross函数，判断RSI_long_term是否上穿SMA_short_term
            df_sma_cross = dfunc.check_cross(
                sma_short_term, rsi_long_term, up_down=True
            )
            # 取上穿SMA_short_term的日期
            up_sma_date = df_sma_cross["上穿"]
            # 表示上穿日期的rsi值
            up_sma_rsi = rsi_long_term[up_sma_date]
            # 期望值设定为 (100-up_sma_rsi)/100
            df_rsi_judge.loc[up_sma_date, period] = round((100 - up_sma_rsi) / 100, 3)

            # 调用check_cross函数，判断RSI_long_term是否下穿SMA_short_term
            df_sma_cross = dfunc.check_cross(
                sma_short_term, rsi_long_term, up_down=True
            )
            # 取下穿SMA_short_term的日期
            down_sma_date = df_sma_cross["下穿"]
            # 表示下穿日期的rsi值
            down_sma_rsi = rsi_long_term[down_sma_date]
            # 期望值设定为 -down_sma_rsi/100
            df_rsi_judge.loc[down_sma_date, period] = round(-down_sma_rsi / 100, 3)

        # 保存策略
        super().save_strategy(
            df_judge=df_rsi_judge,
            func_name=MyRSI._versus_sma_strategy.__name__,
            strategy_config_value_dict=None,
            saperate=True,
        )

        return df_rsi_judge


if __name__ == "__main__":
    MyRSI(None, dt.date.today(), "002230", ProductType.Stock, None).analyze()
