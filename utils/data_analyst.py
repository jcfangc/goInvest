"""调用indicator文件夹中的技术指标函数，综合分析，给出买入卖出建议，输出文本、图像到data文件夹中"""

import datetime as dt
import pandas as pd

from pandas import DataFrame as df
from goInvest.utils.indicator import myBoll as mb
from utils import dataSource_picker as dp
from typing import Literal, cast
from . import goInvest_path
from datetime import timedelta


class BollAnalyst:
    """
    根据布林线判断买入卖出\n
    today_date: dt.date,\n
    product_code: str,\n
    product_type: Literal["stock"],
    """

    def __init__(
        self,
        today_date: dt.date,
        product_code: str,
        product_type: Literal["stock"],
    ) -> None:
        self.product_code = product_code
        self.today_date = today_date
        self.product_type = product_type

    def analyze(self) -> list[df]:
        # 计算股票的布林线
        dict_boll = dp.dataPicker.indicator_source_picker(
            product_code=self.product_code,
            today_date=self.today_date,
            indicator_name="Boll",
            product_type=cast(Literal["stock"], self.product_type),
        )

        boll_time_window = 0
        # 找到包含‘中轴’的列名
        for col in dict_boll["daily"].columns:
            if "中轴" in col:
                # 从列名中提取时间窗口（数字）
                boll_time_window = int(col[2:])
                break

        # 获取日K/周K数据
        ohlc_data = dp.dataPicker.product_source_picker(
            product_code=self.product_code,
            today_date=self.today_date,
            product_type=cast(Literal["stock"], self.product_type),
        )

        # 创建一个空的DataFrame
        # 第一列为“日期”索引，第二列（daily）第三列（weekly）为-1至1的布林判断值
        df_boll_judge = df(index=ohlc_data["daily"].index, columns=["daily", "weekly"])
        # 初始化布林线判断值为0
        df_boll_judge["daily"] = 0
        df_boll_judge["weekly"] = 0

        # 从字典中取出布林线数据
        for period in ["daily", "weekly"]:
            # 按照日期顺序遍历
            for date in ohlc_data[period].index:
                # 从ohlc_data中取出对应日期的数据
                ohlc_data_date = ohlc_data[period].loc[date]  # 包含开盘、收盘、最高、最低、成交量……
                # 检测dict_boll中的值是否非空
                if any(df.empty for df in dict_boll.values()):
                    raise ValueError("布林线数据为空！")
                else:
                    # 从dict_boll中取出对应日期的布林线数据
                    boll_data_date = dict_boll[period].loc[date]  # 包含上轨、中轴、下轨

                # 如果蜡烛线的最高值和最低值的差大于上轨线和下轨线的差，说明蜡烛线的波动幅度大于布林线的波动幅度，不适合用布林线判断
                if (
                    ohlc_data_date["最高"] - ohlc_data_date["最低"]
                    > boll_data_date["上轨"] - boll_data_date["下轨"]
                ):
                    continue

                # 需要注意的是都要从最严重的情况开始分析数值，即先判断±1，再判断±0.8，再判断±0.6，...
                # 如果开盘价和收盘价的均值大于中轴线，只围绕上轨线分析
                if (ohlc_data_date["开盘"] + ohlc_data_date["收盘"]) / 2 >= boll_data_date[
                    f"中轴{boll_time_window}"
                ]:
                    # 下跌趋势
                    if ohlc_data_date["收盘"] < ohlc_data_date["开盘"]:
                        # 1. 下影线大于上轨线，df_boll_judge的值为-1
                        if ohlc_data_date["最低"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -1
                        # 2. 收盘价大于上轨线，df_boll_judge的值为-0.95
                        elif ohlc_data_date["收盘"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.95
                        # 3. 开盘价和收盘价的均值大于上轨线，df_boll_judge的值为-0.75
                        elif (
                            ohlc_data_date["开盘"] + ohlc_data_date["收盘"]
                        ) / 2 >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.75
                        # 4. 开盘价大于上轨线，df_boll_judge的值为-0.5
                        elif ohlc_data_date["开盘"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.5
                        # 5. 上影线大于上轨线，df_boll_judge的值为-0.25
                        elif ohlc_data_date["最高"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.25

                    # 上涨趋势
                    elif ohlc_data_date["收盘"] >= ohlc_data_date["开盘"]:
                        # 1. 下影线大于上轨线，df_boll_judge的值为-1
                        if ohlc_data_date["最低"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -1
                        # 2. 开盘价大于上轨线，df_boll_judge的值为-0.95
                        elif ohlc_data_date["开盘"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.95
                        # 3. 开盘价和收盘价的均值大于上轨线，df_boll_judge的值为-0.75
                        elif (
                            ohlc_data_date["开盘"] + ohlc_data_date["收盘"]
                        ) / 2 >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.75
                        # 4. 收盘价大于上轨线，df_boll_judge的值为-0.5
                        elif ohlc_data_date["收盘"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.5
                        # 5. 上影线大于上轨线，df_boll_judge的值为-0.25
                        elif ohlc_data_date["最高"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.25

                    # 横盘趋势
                    else:
                        # 1. 下影线大于上轨线，df_boll_judge的值为-1
                        if ohlc_data_date["最低"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -1
                        # 2. 收盘价等于开盘价大于上轨线，df_boll_judge的值为-0.75
                        elif ohlc_data_date["收盘"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.75
                        # 3. 上影线大于上轨线，df_boll_judge的值为-0.25
                        elif ohlc_data_date["最高"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.25

                # 如果开盘价和收盘价的均值小于中轴线，只围绕下轨线分析
                elif (ohlc_data_date["开盘"] + ohlc_data_date["收盘"]) / 2 < boll_data_date[
                    f"中轴{boll_time_window}"
                ]:
                    # 下跌趋势
                    if ohlc_data_date["收盘"] < ohlc_data_date["开盘"]:
                        # 10. 上影线小于下轨线，df_boll_judge的值为1
                        if ohlc_data_date["最高"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 1
                        # 9. 开盘价小于下轨线，df_boll_judge的值为0.8
                        elif ohlc_data_date["开盘"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 0.8
                        # 8. 开盘价和收盘价的均值小于下轨线，df_boll_judge的值为0.6
                        elif (
                            ohlc_data_date["开盘"] + ohlc_data_date["收盘"]
                        ) / 2 <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 0.6
                        # 7. 收盘价小于下轨线，df_boll_judge的值为0.4
                        elif ohlc_data_date["收盘"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 0.4
                        # 6. 下影线小于下轨线，df_boll_judge的值为0.2
                        elif ohlc_data_date["最低"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 0.2

                    # 上涨趋势
                    elif ohlc_data_date["收盘"] > ohlc_data_date["开盘"]:
                        # 10. 上影线小于下轨线，df_boll_judge的值为1
                        if ohlc_data_date["最高"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 1
                        # 9. 收盘价小于下轨线，df_boll_judge的值为0.8
                        elif ohlc_data_date["收盘"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 0.8
                        # 8. 开盘价和收盘价的均值小于下轨线，df_boll_judge的值为0.6
                        elif (
                            ohlc_data_date["开盘"] + ohlc_data_date["收盘"]
                        ) / 2 <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 0.6
                        # 7. 开盘价小于下轨线，df_boll_judge的值为0.4
                        elif ohlc_data_date["开盘"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 0.4
                        # 6. 下影线小于下轨线，df_boll_judge的值为0.2
                        elif ohlc_data_date["最低"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 0.2

                    # 横盘趋势
                    else:
                        # 3. 上影线大于上轨线，df_boll_judge的值为-0.2
                        if ohlc_data_date["最高"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 1
                        # 2. 收盘价等于开盘价大于上轨线，df_boll_judge的值为-0.6
                        elif ohlc_data_date["收盘"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 0.6
                        # 1. 下影线大于上轨线，df_boll_judge的值为-1
                        elif ohlc_data_date["最低"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 0.2

        # 输出布林线判断结果为csv，放在data\\kline\\strategy文件夹中
        with open(
            file=f"{goInvest_path}\\goInvest\\data\\kline\\indicator\\strategy\\{self.product_code}_Boll_analysis.csv",
            mode="w",
            encoding="utf-8",
        ) as f:
            df_boll_judge.to_csv(f)
            print(f"查看{self.product_code}的布林线分析结果\n>>>>{f.name}")

        return [df_boll_judge]


class MAAnalyst:
    """
    根据移动平均线判断买入卖出\n
    today_date: dt.date,\n
    product_code: str,\n
    product_type: Literal["stock"],
    """

    def __init__(
        self,
        today_date: dt.date,
        product_code: str,
        product_type: Literal["stock"],
    ) -> None:
        self.product_code = product_code
        self.today_date = today_date
        self.product_type = product_type

    def analyze(self) -> list[df]:
        # 获取股票K线移动平均线数据
        dict_sma_data = dp.dataPicker.indicator_source_picker(
            product_code=self.product_code,
            today_date=self.today_date,
            product_type=cast(Literal["stock"], self.product_type),
            indicator_name="SMA",
        )

        if any(df.empty for df in dict_sma_data.values()):
            raise ValueError("移动平均线数据为空！")

        # 调用策略函数
        sma_elastic_band_judge = self._elastic_band_strategy(dict_sma_data, "SMA")
        sma_mutiline_judge = self._mutiline_strategy(dict_sma_data, "SMA")

        # 获取股票K线指数移动平均线数据
        dict_ema_data = dp.dataPicker.indicator_source_picker(
            product_code=self.product_code,
            today_date=self.today_date,
            product_type=cast(Literal["stock"], self.product_type),
            indicator_name="EMA",
        )

        if any(df.empty for df in dict_ema_data.values()):
            raise ValueError("指数移动平均线数据为空！")

        # 调用策略函数
        ema_elastic_band_judge = self._elastic_band_strategy(dict_ema_data, "EMA")
        ema_mutiline_judge = self._mutiline_strategy(dict_ema_data, "EMA")

        return [
            sma_elastic_band_judge,
            sma_mutiline_judge,
            ema_elastic_band_judge,
            ema_mutiline_judge,
        ]

    def _elastic_band_strategy(
        self, dict_ma_data: dict[str, df], ma_name: Literal["SMA", "EMA"]
    ) -> df:
        """
        弹力带策略\n
        这是一个基于经验的移动平均线策略，不一定具有普适性\n
        策略如下：\n
        1. 150均线>50均线>20均线>5均线\n
        2. 均线之间的差距较为均匀，且差距的比值在某个阈值以上\n
        3. 此时价格是局部较低点（山谷）\n
        由于数据的滞后性，判断出山谷时，已经错过了最佳买入时机\n
        因此，需要在山谷之后的上升趋势中卖出\n
        """
        # 创建一个空的DataFrame
        # 第一列为“日期”索引，第二列（daily）第三列（weekly）为-1至1的SMA判断值
        df_ma_judge = df(index=dict_ma_data["daily"].index, columns=["daily", "weekly"])
        # 初始化布林线判断值为0
        df_ma_judge["daily"] = 0
        df_ma_judge["weekly"] = 0

        for period in dict_ma_data.keys():
            # print(f"正在计算'{period}'移动平均线弹力带策略...")
            # 创建副本
            new_dict_ma_data = dict_ma_data.copy()
            # 保留dict_sma_data[period]的中150均线>50均线>20均线>5均线的行
            new_dict_ma_data[period] = new_dict_ma_data[period][
                (new_dict_ma_data[period]["150均线"] > new_dict_ma_data[period]["50均线"])
                & (new_dict_ma_data[period]["50均线"] > new_dict_ma_data[period]["20均线"])
                & (new_dict_ma_data[period]["20均线"] > new_dict_ma_data[period]["5均线"])
            ]

            # 判断条件1和条件2
            # 创建一个空的DataFrame
            ratio_df = df(
                columns=[
                    "differ_1",
                    "differ_2",
                    "differ_3",
                    "ratio_1",
                    "ratio_2",
                    "ratio_3",
                    "avg_ratio",
                ],
                # 第一列为“日期”索引
                index=new_dict_ma_data[period].index,
            )
            # 第二列differ_1(150,50)
            ratio_df["differ_1"] = (
                new_dict_ma_data[period]["150均线"] - new_dict_ma_data[period]["50均线"]
            ).round(3)
            # 第三列differ_2(50,20)
            ratio_df["differ_2"] = (
                new_dict_ma_data[period]["50均线"] - new_dict_ma_data[period]["20均线"]
            ).round(3)
            # 第四列differ_3(20,5)
            ratio_df["differ_3"] = (
                new_dict_ma_data[period]["20均线"] - new_dict_ma_data[period]["5均线"]
            ).round(3)
            # 第五列ratio_1(differ_1/differ_2)
            ratio_df["ratio_1"] = (ratio_df["differ_1"] / ratio_df["differ_2"]).round(4)
            # 第六列ratio_2(differ_2/differ_3)
            ratio_df["ratio_2"] = (ratio_df["differ_2"] / ratio_df["differ_3"]).round(4)
            # 第七列ratio_3(differ_1/differ_3)
            ratio_df["ratio_3"] = (ratio_df["differ_1"] / ratio_df["differ_3"]).round(4)
            # ratio_1和ratio_2和ratio_3中大于1的值取倒数
            for col in ["ratio_1", "ratio_2", "ratio_3"]:
                # ratio_df[col] > 1 返回的是bool值，ratio_df.loc[ratio_df[col] > 1, col]返回的是大于1的值
                ratio_df.loc[ratio_df[col] > 1, col] = (1 / ratio_df[col]).round(3)
            # 第八列avg_ratio(ratio_1+ratio_2+ratio_3)/3
            ratio_df["avg_ratio"] = (
                (ratio_df["ratio_1"] + ratio_df["ratio_2"] + ratio_df["ratio_3"]) / 3
            ).round(3)

            # 保留ratio_df中avg_ratio>=阈值的行
            ratio_df = ratio_df[ratio_df["avg_ratio"] >= 0.5]  # 阈值为0.5
            # print(f"ratio_df:\n{ratio_df}")

            # 判断条件3
            # 创建一个空的DataFrame
            valley_df = df(
                columns=["5均线", "is_valley"],
                # 第一列为“日期”索引
                index=new_dict_ma_data[period].index,
            )
            # 第二列new_dict_sma_data[period]["5均线"]的值
            valley_df["5均线"] = new_dict_ma_data[period]["5均线"]
            # 重新设置index为整数，将日期作为一列
            valley_df.reset_index(inplace=True)
            # 第三列是布尔值，初始化为False
            valley_df["is_valley"] = False
            # 某个日期的5均线值同时小于五天前和五天后的5均线值，返回True，否则返回False
            for index, _ in valley_df.iterrows():
                # 保证index在valid_df的index范围内
                if (index > 19) and (valley_df.loc[index - 5, "is_valley"] == False):  # type: ignore
                    if (
                        valley_df.loc[index - 5, "5均线"]  # type: ignore
                        < valley_df.loc[index - 10, "5均线"]  # type: ignore
                    ) & (
                        valley_df.loc[index - 5, "5均线"] < valley_df.loc[index, "5均线"]  # type: ignore
                    ):
                        valley_df.loc[index - 5, "is_valley"] = True  # type: ignore
                    else:
                        valley_df.loc[index - 5, "is_valley"] = False  # type: ignore

            # 重新设置index为日期
            valley_df.set_index("日期", inplace=True)
            # 保留is_end_of_valley=True的行
            valley_df = valley_df[valley_df["is_valley"] == True]
            # print(f"valley_df:\n{valley_df}")

            # 求出valley_df和ratio_df的日期交集
            intersection_date = []
            # 只要ratio_df和valley_df的日期index相差的绝对值小于等于5，就认为是交集
            for date1 in ratio_df.index:
                for date2 in valley_df.index:
                    # 如果date1和date2已经在intersection_date中，就跳过
                    if date2 in intersection_date:
                        continue
                    if (
                        (abs(date1 - date2) <= timedelta(days=5))
                        if period == "daily"
                        else (abs(date1 - date2) <= timedelta(weeks=5))
                    ):
                        if date1 not in intersection_date:
                            intersection_date.append(date1)
                        intersection_date.append(date2)

            # 加入df_sma_judge最后的index
            intersection_date.append(df_ma_judge.index[-1])
            # 整理intersection_date顺序
            intersection_date.sort()
            # 连续的日期中（日期相差小于三日视作连续）取最早的一个，其余删除
            pop_index = []
            for i in range(len(intersection_date) - 1):
                difference = abs(intersection_date[i] - intersection_date[i + 1])
                if (
                    (difference <= timedelta(days=3))
                    if period == "daily"
                    else (difference <= timedelta(weeks=3))
                ):
                    pop_index.append(i + 1)
            # 从后往前删除
            for i in pop_index[::-1]:
                intersection_date.pop(i)
            # print(f"intersection_date:\n{intersection_date}")

            # 创建空的DataFrame
            find_sold_df = df(
                index=dict_ma_data[period].index,
                columns=["5均线-10均线", "5均线-20均线", "5均线-50均线"],
            )
            # 得到相应数值
            find_sold_df["5均线-10均线"] = (
                dict_ma_data[period]["5均线"] - dict_ma_data[period]["10均线"]
            ).round(3)
            find_sold_df["5均线-20均线"] = (
                dict_ma_data[period]["5均线"] - dict_ma_data[period]["20均线"]
            ).round(3)
            find_sold_df["5均线-50均线"] = (
                dict_ma_data[period]["5均线"] - dict_ma_data[period]["50均线"]
            ).round(3)

            # # 打印各个df日期，用于调试
            # print(f"intersection_date:\n{intersection_date}")
            # print(f"find_sold_df:\n{find_sold_df.index}")
            # print(f"df_ma_judge:\n{df_ma_judge.index}")

            # intersect_index是山谷的日期，山谷之后出现的均线上穿现象，都应该是卖出时机
            # 用for循环遍历dates
            for date in intersection_date[1:]:
                first_10_signal = False
                first_20_signal = False
                first_50_signal = False
                # print(
                #     f"\nform {intersection_date[intersection_date.index(date) - 1]} to {date}"
                # )
                # 遍历本循环的date和上个循环的date之间的日期
                for index in find_sold_df.index[
                    df_ma_judge.index.get_loc(
                        intersection_date[intersection_date.index(date) - 1]
                    ) : df_ma_judge.index.get_loc(date)
                ]:
                    # 如果第一次的上穿现象都检测到了，就跳出循环
                    if (
                        first_10_signal and first_20_signal and first_50_signal
                    ) == True:
                        break
                    # 此后遇到的第一个5均线值大于10均线值的日期，将其买卖值设为-0.6，应该卖出
                    if (
                        find_sold_df.loc[index, "5均线-10均线"] >= 0  # type: ignore
                        and first_10_signal == False
                    ):
                        # print("5均线-10均线：" + f"{find_sold_df.loc[index, '5均线-10均线']}")
                        # print(index)
                        df_ma_judge.loc[index, period] = -0.6
                        first_10_signal = True
                    # 此后遇到的第一个5均线值大于10均线值的日期，将其买卖值设为-0.8，应该卖出
                    if (
                        find_sold_df.loc[index, "5均线-20均线"] >= 0  # type: ignore
                        and first_20_signal == False
                    ):
                        # print("5均线-20均线：" + f"{find_sold_df.loc[index, '5均线-20均线']}")
                        # print(index)
                        df_ma_judge.loc[index, period] = -0.8
                        first_20_signal = True
                    # 此后遇到的第一个5均线值大于50均线值的日期，将其买卖值设为-1，应该卖出
                    if (
                        find_sold_df.loc[index, "5均线-50均线"] >= 0  # type: ignore
                        and first_50_signal == False
                    ):
                        # print("5均线-50均线：" + f"{find_sold_df.loc[index, '5均线-50均线']}")
                        # print(index)
                        df_ma_judge.loc[index, period] = -1
                        first_50_signal = True

        # 输出df_sma_judge为csv文件，在strategy文件夹中
        with open(
            file=(
                f"{goInvest_path}\\goInvest\\data\\kline\\indicator\\strategy\\{self.product_code}_{ma_name}_elastic_band_anlysis.csv"
            ),
            mode="w",
            encoding="utf-8",
        ) as f:
            df_ma_judge.to_csv(f)
            print(f"查看{self.product_code}的移动均线（弹力绳策略）分析结果\n>>>>{f.name}")

        return df_ma_judge

    def _mutiline_strategy(
        self, dict_ma_data: dict[str, df], ma_name: Literal["SMA", "EMA"]
    ) -> df:
        """
        多线策略，即多条均线同时作为买入卖出的依据\n
        在一段上涨态势中\n
        如果收盘价格跌破5均线，决策值设为-0.65\n
        如果收盘价格跌破10均线，决策值设为-0.85\n
        如果收盘价格跌破20均线，决策值设为-0.95\n
        如果收盘价格跌破50均线，决策值设为-1\n
        """

        # 创建一个空的DataFrame
        # 第一列为“日期”索引，第二列（daily）第三列（weekly）为-1至1的SMA判断值
        df_ma_judge = df(index=dict_ma_data["daily"].index, columns=["daily", "weekly"])

        # 初始化布林线判断值为0
        df_ma_judge["daily"] = 0
        df_ma_judge["weekly"] = 0

        # 获取股票数据
        ohlc_data = dp.dataPicker.product_source_picker(
            product_code=self.product_code,
            today_date=self.today_date,
            product_type=cast(Literal["stock"], self.product_type),
        )
        # 获取股票数据的日期，作为基准
        # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
        standard_daily_date = ohlc_data["daily"].index
        standard_weekly_date = ohlc_data["weekly"].index

        for period in dict_ma_data.keys():
            ma_data = dict_ma_data[period]
            # print(sma_data.head(50))
            # 上涨态势的判断
            # 5均线大于10均线，10均线大于20均线，20均线大于50均线，50均线大于150均线
            up_trend = ma_data[
                (ma_data["5均线"] > ma_data["10均线"])
                & (ma_data["10均线"] > ma_data["20均线"])
                & (ma_data["20均线"] > ma_data["50均线"])
                & (ma_data["50均线"] > ma_data["150均线"])
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
                        # prev_date之前的20个交易日
                        prev_date_20 = standard_daily_date[
                            standard_daily_date < prev_date
                        ][-20:]
                        # date之后的20个交易日
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
                        # prev_date之前的4个交易日
                        prev_date_4 = standard_weekly_date[
                            standard_weekly_date < prev_date
                        ][-4:]
                        # date之后的4个交易日
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
                close_price = ohlc_data[period].loc[date, "收盘"]
                # 表示50均线
                sma_50 = ma_data.loc[date, "50均线"]
                # 表示20均线
                sma_20 = ma_data.loc[date, "20均线"]
                # 表示10均线
                sma_10 = ma_data.loc[date, "10均线"]
                # 表示5均线
                sma_5 = ma_data.loc[date, "5均线"]

                # 如果收盘价跌破50均线，决策值设为-1
                if close_price <= sma_50:  # type: ignore
                    df_ma_judge.loc[date, period] = -1
                # 如果收盘价跌破20均线，决策值设为-0.9
                elif close_price <= sma_20:  # type: ignore
                    df_ma_judge.loc[date, period] = -0.95
                # 如果收盘价跌破10均线，决策值设为-0.75
                elif close_price <= sma_10:  # type: ignore
                    df_ma_judge.loc[date, period] = -0.85
                # 如果收盘价跌破5均线，决策值设为-0.5
                elif close_price <= sma_5:  # type: ignore
                    df_ma_judge.loc[date, period] = -0.65
                else:
                    continue

        # 输出df_sma_judge为csv文件，在strategy文件夹中
        with open(
            f"{goInvest_path}\\goInvest\\data\\kline\\indicator\\strategy\\{self.product_code}_{ma_name}_multiline_anlysis.csv",
            "w",
            encoding="utf-8",
        ) as f:
            df_ma_judge.to_csv(f)
            print(f"查看{self.product_code}的移动均线（多均线策略）分析结果\n>>>>{f.name}")

        return df_ma_judge


class RSIAnalyst:
    """
    根据RSI指标判断买入卖出\n
    today_date: dt.date,\n
    product_code: str,\n
    product_type: Literal["stock"],\n
    """

    def __init__(
        self,
        today_date: dt.date,
        product_code: str,
        product_type: Literal["stock"],
    ) -> None:
        self.product_code = product_code
        self.today_date = today_date
        self.product_type = product_type

    def analyze(self) -> list[df]:
        # 获取股票K线RSI数据
        dict_rsi_data = dp.dataPicker.indicator_source_picker(
            product_code=self.product_code,
            today_date=self.today_date,
            product_type=cast(Literal["stock"], self.product_type),
            indicator_name="RSI",
        )

        if any(df.empty for df in dict_rsi_data.values()):
            raise ValueError("RSI数据为空！")

        # 调用策略函数

        return []

    def _over_bought_sold_strategy(self, dict_rsi_data: dict[str, df]) -> df:
        """
        当RSI指标超买超卖时，买入卖出\n
        超买：RSI指标大于70\n
        超卖：RSI指标小于30\n
        """
        # 创建一个空的DataFrame，用于存放买入卖出数值
        df_rsi_judge = df(
            index=dict_rsi_data["daily"].index, columns=["daily", "weekly"]
        )
        # 初始化买入卖出数值为0.0
        df_rsi_judge["daily"] = 0.0
        df_rsi_judge["weekly"] = 0.0

        # 遍历dict_rsi_data["daily"]的列，找到包含字符串“RSI_”的两列名，提取列名中的数字
        short_term = long_term = 0
        for col in dict_rsi_data["daily"].columns:
            if "RSI_" in col:
                if short_term == 0:
                    short_term = str(col).split("_")[1]
                else:
                    long_term = str(col).split("_")[1]

        if short_term == 0 or long_term == 0:
            raise ValueError("RSI数据中缺少包含字符串“RSI_”的列！")
        elif short_term >= long_term:
            # 交换short_term和long_term的值
            short_term, long_term = long_term, short_term

        return df_rsi_judge


class StockAnalyst(BollAnalyst, MAAnalyst):
    """
    传入股票K线数据，分析后返回买入卖出建议
    """

    def __init__(self, stock_code: str, today_date: dt.date) -> None:
        self.product_code = stock_code
        self.today_date = today_date
        self.product_type = "stock"

    def analyze(self):
        # 调用BollAnalyst类的analyze方法
        self.result_list = BollAnalyst.analyze(self)
        # 调用SMAAnalyst类的analyze方法
        self.result_list.extend(MAAnalyst.analyze(self))

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
            file=f"{goInvest_path}\\goInvest\\data\\kline\\indicator\\strategy\\{self.product_code}_comprehensive_anlysis.csv",
            mode="w",
            encoding="utf-8",
        ) as f:
            df_comperhensive_judge.to_csv(f)
            print(f"查看{self.product_code}的综合分析结果\n>>>>{f.name}")
