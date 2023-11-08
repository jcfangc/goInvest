"""myEMA.py"""

if __name__ == "__main__":
    import sys
    import os

    # 将上级目录加入sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))


import datetime as dt
import pandas as pd

from pandas import DataFrame
from utils.myIndicator_abc import MyIndicator
from utils.enumeration_label import ProductType, IndicatorName
from typing import Optional

# 本指标的参数
params = {}


class MyEMA(MyIndicator):
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
            data_path=data_path, indicator_name=IndicatorName.EMA, **params
        )

    def calculate_indicator(self) -> dict[str, DataFrame]:
        """
        本函数根据时间窗口计算移动平均线
        """
        print(f"正在计算{self.product_code}指数移动平均线...")

        # 清理重复文件
        super()._remove_redundant_files()

        # 本指标的参数
        default_indicator_config_value = {"span_list": [5, 10, 20, 50, 150]}
        indicator_config_value = (
            super().read_from_config(None) or default_indicator_config_value
        )

        # 定义一个字典，用于存放返回的不同周期ema数据
        df_ema_dict = {
            "daily": DataFrame(),
            "weekly": DataFrame(),
        }

        span_list = indicator_config_value["span_list"]
        # 从小到大排序
        span_list.sort()
        # 根据数据和时间窗口滚动计算ema
        for period in ["daily", "weekly"]:
            closing_price = self.product_df_dict[period]["收盘"]
            # 5时间窗口，数值取三位小数
            ema_5 = closing_price.ewm(span=span_list[0], adjust=False).mean()
            # 10时间窗口，数值取三位小数
            ema_10 = closing_price.ewm(span=span_list[1], adjust=False).mean()
            # 20时间窗口，数值取三位小数
            ema_20 = closing_price.ewm(span=span_list[2], adjust=False).mean()
            # 50时间窗口，数值取三位小数
            ema_50 = closing_price.ewm(span=span_list[3], adjust=False).mean()
            # 150时间窗口，数值取三位小数
            ema_150 = closing_price.ewm(span=span_list[4], adjust=False).mean()

            # 均线数据汇合
            df_ema_dict[period] = DataFrame(
                {
                    # 日期作为索引
                    # 均线数据
                    "5均线": ema_5.round(3),
                    "10均线": ema_10.round(3),
                    "20均线": ema_20.round(3),
                    "50均线": ema_50.round(3),
                    "150均线": ema_150.round(3),
                }
            )

        # 保存指标
        super().save_indicator(
            df_dict=df_ema_dict, indicator_value_config_dict=indicator_config_value
        )
        # 返回字典
        return df_ema_dict

    def analyze(self) -> list[DataFrame]:
        # 获取股票K线EMA数据
        dict_ema = super().get_dict()
        # 调用策略函数
        ema_mutiline_judge = self._mutiline_strategy(dict_ema)
        # 返回策略
        return [ema_mutiline_judge]

    # 策略函数名请轻易不要修改！！！若修改，需要同时修改枚举类内的StrategyName！！！
    def _mutiline_strategy(
        self,
        dict_ema: dict[str, DataFrame],
    ) -> DataFrame:
        """
        多线策略，即多条均线同时作为判断上涨下跌的依据\n
        在一段上涨态势中\n
        如果收盘价格跌破5均线，决策值设为-0.65\n
        如果收盘价格跌破10均线，决策值设为-0.85\n
        如果收盘价格跌破20均线，决策值设为-0.95\n
        如果收盘价格跌破50均线，决策值设为-1\n
        """

        # 策略参数
        default_strategy_config_value = {"expect_list": [-1, -0.95, -0.85, -0.65]}
        strategy_config_value = (
            super().read_from_config(MyEMA._mutiline_strategy.__name__)
            or default_strategy_config_value
        )

        # 创建一个空的DataFrame
        # 第一列为“日期”索引，第二列（daily）第三列（weekly）为-1至1的ema判断值
        df_ema_judge = DataFrame(
            index=dict_ema["daily"].index, columns=["daily", "weekly"]
        )

        # 初始化布林线判断值为0
        df_ema_judge["daily"] = 0
        df_ema_judge["weekly"] = 0

        # 获取股票数据
        ohlc_data = self.product_df_dict

        # 获取股票数据的日期，作为基准
        # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
        standard_daily_date = ohlc_data["daily"].index
        standard_weekly_date = ohlc_data["weekly"].index

        for period in dict_ema.keys():
            ema_data = dict_ema[period]
            # print(ema_data.head(50))
            # 上涨态势的判断
            # 5均线大于10均线，10均线大于20均线，20均线大于50均线，50均线大于150均线
            up_trend = ema_data[
                (ema_data.iloc[:, 0] > ema_data.iloc[:, 1])
                & (ema_data.iloc[:, 1] > ema_data.iloc[:, 2])
                & (ema_data.iloc[:, 2] > ema_data.iloc[:, 3])
                & (ema_data.iloc[:, 3] > ema_data.iloc[:, 4])
            ].index
            # 确保up_trend是时间格式
            up_trend = pd.to_datetime(up_trend)

            up_trend_too = []
            # 遍历上涨态势的日期
            for i in range(1, len(up_trend)):
                date = up_trend[i]
                prev_date = up_trend[i - 1]
                # print(f"\ndate: {date}, prev_date: {prev_date}")
                # 如果日期前后间隔小于5个交易日/2个交易周
                # 那么将date和prev_date之间的日期视作连续的上升区间
                if period == "daily":
                    # 计算得出standard_daily_date中，prev_date和date之间的日期有几个（不包括prev_date和date）
                    mask = (standard_daily_date > prev_date) & (
                        standard_daily_date < date
                    )
                    dates_chosen = standard_daily_date[mask]
                    # print(dates_between) if len(dates_between) > 0 else None
                    if len(dates_chosen) <= 5 and len(dates_chosen) > 0:
                        # prev_date之前的20个交易日（一个月）
                        prev_date_20 = standard_daily_date[
                            standard_daily_date < prev_date
                        ][-20:]
                        # date之后的20个交易日（一个月）
                        date_20 = standard_daily_date[standard_daily_date > date][:20]
                        # 添加整个列表而非单个元素就用extend
                        up_trend_too.extend(dates_chosen)
                        up_trend_too.extend(prev_date_20)
                        up_trend_too.extend(date_20)
                elif period == "weekly":
                    # 计算得出standard_weekly_date中，prev_date和date之间的日期有几个（不包括prev_date和date）
                    mask = (standard_weekly_date > prev_date) & (
                        standard_weekly_date < date
                    )
                    dates_chosen = standard_weekly_date[mask]
                    # print(dates_between) if len(dates_between) > 0 else None
                    if len(dates_chosen) <= 2 and len(dates_chosen) > 0:
                        # prev_date之前的4个交易周
                        prev_date_4 = standard_weekly_date[
                            standard_weekly_date < prev_date
                        ][-4:]
                        # date之后的4个交易周
                        date_4 = standard_weekly_date[standard_weekly_date > date][:4]
                        # 添加整个列表而非单个元素就用extend
                        up_trend_too.extend(dates_chosen)
                        up_trend_too.extend(prev_date_4)
                        up_trend_too.extend(date_4)
                else:
                    continue

            # up_trend去重
            up_trend = list(set(up_trend))
            # up_trend和up_trend_too的合并
            up_trend.extend(up_trend_too)
            up_trend.sort()
            # 将up_trend转换为时间格式
            up_trend = pd.to_datetime(up_trend)
            # print(f"up_trend:\n{up_trend}")

            # 遍历上涨态势的日期
            for date in up_trend:
                # 表示收盘价
                close_price = ohlc_data[period]["收盘"].loc[date]
                # 表示50均线
                ema_50 = ema_data.iloc[:, 3].loc[date]
                # 表示20均线
                ema_20 = ema_data.iloc[:, 2].loc[date]
                # 表示10均线
                ema_10 = ema_data.iloc[:, 1].loc[date]
                # 表示5均线
                ema_5 = ema_data.iloc[:, 0].loc[date]

                expect_list = strategy_config_value["expect_list"]
                # 如果收盘价跌破50均线，决策值设为-1
                if close_price <= ema_50:  # type: ignore
                    df_ema_judge.loc[date, period] = expect_list[0]  # config
                # 如果收盘价跌破20均线，决策值设为-0.95
                elif close_price <= ema_20:  # type: ignore
                    df_ema_judge.loc[date, period] = expect_list[1]  # config
                # 如果收盘价跌破10均线，决策值设为-0.85
                elif close_price <= ema_10:  # type: ignore
                    df_ema_judge.loc[date, period] = expect_list[2]  # config
                # 如果收盘价跌破5均线，决策值设为-0.65
                elif close_price <= ema_5:  # type: ignore
                    df_ema_judge.loc[date, period] = expect_list[3]  # config
                else:
                    continue

        # 保存策略
        super().save_strategy(
            df_judge=df_ema_judge,
            func_name=MyEMA._mutiline_strategy.__name__,
            strategy_config_value_dict=strategy_config_value,
        )
        # 返回策略
        return df_ema_judge


if __name__ == "__main__":
    # 调用函数stock_EMA
    MyEMA(None, dt.date.today(), "002230", ProductType.Stock, None).analyze()
