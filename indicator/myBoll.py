"""myBoll.py"""

if __name__ == "__main__":
    import sys
    import os

    # 将上级目录加入sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

import datetime as dt


from pandas import DataFrame
from utils.myIndicator_abc import MyIndicator
from utils import dataSource_picker as dp
from utils.enumeration_label import ProductType, IndicatorName
from typing import Optional

# 用于存放参数
params = {}


class MyBoll(MyIndicator):
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
            indicator_name=IndicatorName.Boll, data_path=data_path, **params
        )

    def calculate_indicator(self) -> dict[str, DataFrame]:
        """
        本函数计算布林带，返回一个字典，包含不同周期的布林带数据
        """
        print(f"正在计算{self.product_code}的布林带...")

        # 清理重复文件
        super()._remove_redundant_files()

        # 本指标的参数
        default_indicator_config_value = {"time_window": 10, "k": 2}
        indicator_config_value = (
            super().read_from_config(None) or default_indicator_config_value
        )

        # 定义布林带的时间窗口
        time_window = indicator_config_value["time_window"]  # config

        # 定义一个字典，用于存放返回的不同周期bollinger数据
        df_bollinger_dict = {
            "daily": DataFrame(),
            "weekly": DataFrame(),
        }

        dict_df_sma = dp.dataPicker.indicator_source_picker(
            indicator_name=IndicatorName.SMA,
            **params,
        )

        for period in dict_df_sma.keys():
            # 获取移动平均线中间SMA
            mid_sma_series = dict_df_sma[period][
                f"{time_window}均线"
            ]  # Name: 10均线, dtype: float64, index.name: 日期

            # 数据对齐
            stock_close_series = self.product_df_dict[period][
                "收盘"
            ]  # Name: 收盘, dtype: float64, index.name: 日期
            # 计算标准差
            sigma = (
                stock_close_series.rolling(time_window).std() * 0.9485
            )  # 根据实际情况挑选最适合的修正系数0.9485

            # 根据实际情况挑选最适合的系数
            k = indicator_config_value["k"]  # config

            # 根据中轴线，求出轨线
            # 上轨线
            up_track = mid_sma_series + k * sigma
            # 中轴线
            mid_axis = mid_sma_series
            # 下轨线
            down_track = mid_sma_series - k * sigma

            # 数据汇合
            df = DataFrame(
                {
                    # 日期作为索引
                    # 布林带数据
                    "上轨": up_track.round(3),
                    f"中轴{time_window}": mid_axis.round(3),
                    "下轨": down_track.round(3),
                }
            )
            # 存入字典
            df_bollinger_dict[period] = df

        # 保存指标
        super().save_indicator(
            df_dict=df_bollinger_dict,
            indicator_value_config_dict=indicator_config_value,
        )
        # 返回字典
        return df_bollinger_dict

    def analyze(self) -> list[DataFrame]:
        # 获取股票K线boll数据
        dict_boll = super().get_dict()
        # 调用策略函数
        sma_mutiline_judge = self._boll_strategy(dict_boll)
        # 返回策略
        return [sma_mutiline_judge]

    # 策略函数名请轻易不要修改！！！若修改，需要同时修改枚举类内的StrategyName！！！
    def _boll_strategy(
        self,
        dict_boll: dict[str, DataFrame],
    ) -> DataFrame:
        # 策略参数
        default_strategy_config_value = {
            "up_track_value_list": [-1, -0.95, -0.75, -0.5, -0.25],
            "down_track_value_list": [1, 0.8, 0.6, 0.4, 0.2],
        }
        strategy_config_value = (
            super().read_from_config(MyBoll._boll_strategy.__name__)
            or default_strategy_config_value
        )

        boll_time_window = 0
        # 找到包含‘中轴’的列名
        for col in dict_boll["daily"].columns:
            if "中轴" in col:
                # 从列名中提取时间窗口（数字）
                boll_time_window = int(col[2:])
                break

        # 获取日K/周K数据
        ohlc_data = self.product_df_dict

        # 创建一个空的DataFrame
        # 第一列为“日期”索引，第二列（daily）第三列（weekly）为-1至1的判断值，表示涨跌预期
        df_boll_judge = DataFrame(
            index=ohlc_data["daily"].index, columns=["daily", "weekly"]
        )
        # 初始化布林线判断值为0
        df_boll_judge["daily"] = 0
        df_boll_judge["weekly"] = 0

        # 从字典中取出布林线数据
        for period in ["daily", "weekly"]:
            # 按照日期顺序遍历
            for date in ohlc_data[period].index:
                # 从ohlc_data中取出对应日期的数据
                ohlc_data_date = ohlc_data[period].loc[date]  # 包含开盘、收盘、最高、最低、成交量……

                # 从dict_boll中取出对应日期的布林线数据
                boll_data_date = dict_boll[period].loc[date]  # 包含上轨、中轴、下轨

                # 如果蜡烛线的最高值和最低值的差大于上轨线和下轨线的差，说明蜡烛线的波动幅度大于布林线的波动幅度，不适合用布林线判断
                if (
                    ohlc_data_date["最高"] - ohlc_data_date["最低"]
                    > boll_data_date["上轨"] - boll_data_date["下轨"]
                ):
                    continue

                # 上轨线数值列表
                up_track_value_list = strategy_config_value[
                    "up_track_value_list"
                ]  # config
                # 下轨线数值列表
                down_track_value_list = strategy_config_value[
                    "down_track_value_list"
                ]  # config

                # 需要注意的是都要从最严重的情况开始分析数值，即先判断±1，再判断±0.8，再判断±0.6，...
                # 如果开盘价和收盘价的均值大于中轴线，只围绕上轨线分析
                if (ohlc_data_date["开盘"] + ohlc_data_date["收盘"]) / 2 >= boll_data_date[
                    f"中轴{boll_time_window}"
                ]:
                    # 当日下跌
                    if ohlc_data_date["收盘"] < ohlc_data_date["开盘"]:
                        # 1. 下影线大于上轨线，df_boll_judge的值为-1
                        if ohlc_data_date["最低"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = up_track_value_list[0]
                        # 2. 收盘价大于上轨线，df_boll_judge的值为-0.95
                        elif ohlc_data_date["收盘"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = up_track_value_list[1]
                        # 3. 开盘价和收盘价的均值大于上轨线，df_boll_judge的值为-0.75
                        elif (
                            ohlc_data_date["开盘"] + ohlc_data_date["收盘"]
                        ) / 2 >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = up_track_value_list[2]
                        # 4. 开盘价大于上轨线，df_boll_judge的值为-0.5
                        elif ohlc_data_date["开盘"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = up_track_value_list[3]
                        # 5. 上影线大于上轨线，df_boll_judge的值为-0.25
                        elif ohlc_data_date["最高"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = up_track_value_list[4]

                    # 当日上涨
                    elif ohlc_data_date["收盘"] >= ohlc_data_date["开盘"]:
                        # 1. 下影线大于上轨线，df_boll_judge的值为-1
                        if ohlc_data_date["最低"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = up_track_value_list[0]
                        # 2. 开盘价大于上轨线，df_boll_judge的值为-0.95
                        elif ohlc_data_date["开盘"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = up_track_value_list[1]
                        # 3. 开盘价和收盘价的均值大于上轨线，df_boll_judge的值为-0.75
                        elif (
                            ohlc_data_date["开盘"] + ohlc_data_date["收盘"]
                        ) / 2 >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = up_track_value_list[2]
                        # 4. 收盘价大于上轨线，df_boll_judge的值为-0.5
                        elif ohlc_data_date["收盘"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = up_track_value_list[3]
                        # 5. 上影线大于上轨线，df_boll_judge的值为-0.25
                        elif ohlc_data_date["最高"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = up_track_value_list[4]

                    # 当日横盘
                    else:
                        # 1. 下影线大于上轨线，df_boll_judge的值为-1
                        if ohlc_data_date["最低"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = up_track_value_list[0]
                        # 2. 收盘价等于开盘价大于上轨线，df_boll_judge的值为-0.75
                        elif ohlc_data_date["收盘"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = up_track_value_list[2]
                        # 3. 上影线大于上轨线，df_boll_judge的值为-0.25
                        elif ohlc_data_date["最高"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = up_track_value_list[4]

                # 如果开盘价和收盘价的均值小于中轴线，只围绕下轨线分析
                elif (ohlc_data_date["开盘"] + ohlc_data_date["收盘"]) / 2 < boll_data_date[
                    f"中轴{boll_time_window}"
                ]:
                    # 当日下跌
                    if ohlc_data_date["收盘"] < ohlc_data_date["开盘"]:
                        # 10. 上影线小于下轨线，df_boll_judge的值为1
                        if ohlc_data_date["最高"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = down_track_value_list[0]
                        # 9. 开盘价小于下轨线，df_boll_judge的值为0.8
                        elif ohlc_data_date["开盘"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = down_track_value_list[1]
                        # 8. 开盘价和收盘价的均值小于下轨线，df_boll_judge的值为0.6
                        elif (
                            ohlc_data_date["开盘"] + ohlc_data_date["收盘"]
                        ) / 2 <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = down_track_value_list[2]
                        # 7. 收盘价小于下轨线，df_boll_judge的值为0.4
                        elif ohlc_data_date["收盘"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = down_track_value_list[3]
                        # 6. 下影线小于下轨线，df_boll_judge的值为0.2
                        elif ohlc_data_date["最低"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = down_track_value_list[4]

                    # 当日上涨
                    elif ohlc_data_date["收盘"] > ohlc_data_date["开盘"]:
                        # 10. 上影线小于下轨线，df_boll_judge的值为1
                        if ohlc_data_date["最高"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = down_track_value_list[0]
                        # 9. 收盘价小于下轨线，df_boll_judge的值为0.8
                        elif ohlc_data_date["收盘"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = down_track_value_list[1]
                        # 8. 开盘价和收盘价的均值小于下轨线，df_boll_judge的值为0.6
                        elif (
                            ohlc_data_date["开盘"] + ohlc_data_date["收盘"]
                        ) / 2 <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = down_track_value_list[2]
                        # 7. 开盘价小于下轨线，df_boll_judge的值为0.4
                        elif ohlc_data_date["开盘"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = down_track_value_list[3]
                        # 6. 下影线小于下轨线，df_boll_judge的值为0.2
                        elif ohlc_data_date["最低"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = down_track_value_list[4]

                    # 当日横盘
                    else:
                        # 3. 上影线小于下轨线，df_boll_judge的值为1
                        if ohlc_data_date["最高"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = down_track_value_list[0]
                        # 2. 收盘价等于开盘价小于下轨线，df_boll_judge的值为0.6
                        elif ohlc_data_date["收盘"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = down_track_value_list[2]
                        # 1. 下影线小于下轨线，df_boll_judge的值为0.2
                        elif ohlc_data_date["最低"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = down_track_value_list[4]

        print(strategy_config_value)
        # 保存策略
        super().save_strategy(
            df_judge=df_boll_judge,
            func_name=MyBoll._boll_strategy.__name__,
            strategy_config_value_dict=strategy_config_value,
        )

        # 返回策略
        return df_boll_judge


if __name__ == "__main__":
    # 测试
    MyBoll(None, dt.date.today(), "002230", ProductType.Stock, None).analyze()
