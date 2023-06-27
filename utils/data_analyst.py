"""调用indicator文件夹中的技术指标函数，综合分析，给出买入卖出建议，输出文本、图像到data文件夹中"""

import datetime as dt

from pandas import DataFrame as df
from goInvest.utils.indicator import myBoll as mb
from utils import dataSource_picker as dp
from typing import Literal, cast
from . import goInvest_path


class BollAnalyst:
    """
    根据布林线判断买入卖出\n
    time_window_5: int,\n
    today_date: dt.date,\n
    product_code: str,\n
    product_type: Literal["stock"],
    """

    def __init__(
        self,
        time_window_5: int,
        today_date: dt.date,
        product_code: str,
        product_type: Literal["stock"],
    ) -> None:
        self.time_window_5 = time_window_5
        self.product_code = product_code
        self.today_date = today_date
        self.product_type = product_type

    def analyze(self) -> df:
        boll_time_window = self.time_window_5 * 2  # （日K）10周期时间窗口，大概是半个月
        # 计算股票的布林线
        dict_boll = dp.dataPicker.indicator_source_picker(
            product_code=self.product_code,
            today_date=self.today_date,
            time_window=boll_time_window,
            indicator_name="Boll",
            product_type=cast(Literal["stock"], self.product_type),
        )
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
                        # 2. 收盘价大于上轨线，df_boll_judge的值为-0.8
                        elif ohlc_data_date["收盘"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.8
                        # 3. 开盘价和收盘价的均值大于上轨线，df_boll_judge的值为-0.6
                        elif (
                            ohlc_data_date["开盘"] + ohlc_data_date["收盘"]
                        ) / 2 >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.6
                        # 4. 开盘价大于上轨线，df_boll_judge的值为-0.4
                        elif ohlc_data_date["开盘"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.4
                        # 5. 上影线大于上轨线，df_boll_judge的值为-0.2
                        elif ohlc_data_date["最高"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.2

                    # 上涨趋势
                    elif ohlc_data_date["收盘"] >= ohlc_data_date["开盘"]:
                        # 1. 下影线大于上轨线，df_boll_judge的值为-1
                        if ohlc_data_date["最低"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -1
                        # 2. 开盘价大于上轨线，df_boll_judge的值为-0.8
                        elif ohlc_data_date["开盘"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.8
                        # 3. 开盘价和收盘价的均值大于上轨线，df_boll_judge的值为-0.6
                        elif (
                            ohlc_data_date["开盘"] + ohlc_data_date["收盘"]
                        ) / 2 >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.6
                        # 4. 收盘价大于上轨线，df_boll_judge的值为-0.4
                        elif ohlc_data_date["收盘"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.4
                        # 5. 上影线大于上轨线，df_boll_judge的值为-0.2
                        elif ohlc_data_date["最高"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.2

                    # 横盘趋势
                    else:
                        # 1. 下影线大于上轨线，df_boll_judge的值为-1
                        if ohlc_data_date["最低"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -1
                        # 2. 收盘价等于开盘价大于上轨线，df_boll_judge的值为-0.6
                        elif ohlc_data_date["收盘"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.6
                        # 3. 上影线大于上轨线，df_boll_judge的值为-0.2
                        elif ohlc_data_date["最高"] >= boll_data_date["上轨"]:
                            df_boll_judge.loc[date, period] = -0.2

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

        return df_boll_judge


class SMAAnalyst:
    """
    根据移动平均线判断买入卖出\n
    time_window_5: int,\n
    today_date: dt.date,\n
    product_code: str,\n
    product_type: Literal["stock"],
    """

    def __init__(
        self,
        time_window_5: int,
        today_date: dt.date,
        product_code: str,
        product_type: Literal["stock"],
    ) -> None:
        self.time_window_5 = time_window_5
        self.product_code = product_code
        self.today_date = today_date
        self.product_type = product_type

    def analyze(self):
        # 获取股票K线移动平均线数据
        dict_sma_data = dp.dataPicker.indicator_source_picker(
            product_code=self.product_code,
            today_date=self.today_date,
            time_window=self.time_window_5,
            product_type=cast(Literal["stock"], self.product_type),
            indicator_name="SMA",
        )

        # 调用策略函数
        self._elastic_band_stradegy(dict_sma_data)

    def _elastic_band_stradegy(self, dict_sma_data: dict[str, df]):
        """
        弹力带策略\n
        这是一个基于经验的移动平均线策略，不一定具有普适性\n
        策略如下：\n
        1. 150均线>50均线>20均线>5均线\n
        2. 均线之间的差距较为均匀，且差距的比值在0.8以上\n
        3. 此时价格是局部较低点（山谷）\n
        满足以上条件时，买入\n
        当5均线上穿50均线时，卖出\n
        """

        if any(df.empty for df in dict_sma_data.values()):
            raise ValueError("移动平均线数据为空！")

        # 创建一个空的DataFrame
        # 第一列为“日期”索引，第二列（daily）第三列（weekly）为-1至1的SMA判断值
        df_sma_judge = df(
            index=dict_sma_data["daily"].index, columns=["daily", "weekly"]
        )
        # 初始化布林线判断值为0
        df_sma_judge["daily"] = 0
        df_sma_judge["weekly"] = 0

        for period in dict_sma_data.keys():
            # 去除含有nan的行
            dict_sma_data[period].dropna(axis=0, how="any", inplace=True)
            # 创建副本
            new_dict_sma_data = dict_sma_data.copy()
            # 保留dict_sma_data[period]的中150均线>50均线>20均线>5均线的行
            new_dict_sma_data[period] = new_dict_sma_data[period][
                (dict_sma_data[period]["150均线"] > dict_sma_data[period]["50均线"])
                & (dict_sma_data[period]["50均线"] > dict_sma_data[period]["20均线"])
                & (dict_sma_data[period]["20均线"] > dict_sma_data[period]["5均线"])
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
                index=new_dict_sma_data[period].index,
            )
            # 第二列differ_1(150,50)
            ratio_df["differ_1"] = (
                new_dict_sma_data[period]["150均线"] - new_dict_sma_data[period]["50均线"]
            ).round(3)
            # 第三列differ_2(50,20)
            ratio_df["differ_2"] = (
                new_dict_sma_data[period]["50均线"] - new_dict_sma_data[period]["20均线"]
            ).round(3)
            # 第四列differ_3(20,5)
            ratio_df["differ_3"] = (
                new_dict_sma_data[period]["20均线"] - new_dict_sma_data[period]["5均线"]
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

            # 重置索引
            ratio_df.reset_index(inplace=True)
            # 保留每个满足avg_ratio>=0.8条件之下的第十行到十五行
            sliced_rows = []
            for index, row in ratio_df.iterrows():
                if row["avg_ratio"] >= 0.8:
                    sliced_rows.extend(
                        ratio_df.iloc[index + 10 : index + 15].values.tolist()  # type: ignore
                    )
            sliced_ratio_df = df(sliced_rows, columns=ratio_df.columns)
            # 回复日期索引
            sliced_ratio_df.set_index("日期", inplace=True)

            print(f"{period}ratio：\n{ratio_df.head(10)}")

            # 判断条件3
            # 创建一个空的DataFrame
            vally_df = df(
                columns=["10均线", "is_end_of_vally"],
                # 第一列为“日期”索引
                index=new_dict_sma_data[period].index,
            )
            # 第二列new_dict_sma_data[period]["5均线"]的值
            vally_df["10均线"] = new_dict_sma_data[period]["10均线"]
            # 重新设置index为整数，将日期作为一列
            vally_df.reset_index(inplace=True)
            # 第三列是布尔值，初始化为False
            vally_df["is_end_of_vally"] = False
            # 某个日期的5均线值同时小于五天前和五天后的5均线值，返回True，否则返回False
            for index, _ in vally_df.iterrows():
                # 保证index在valid_df的index范围内
                if (index > 19) and (vally_df.loc[index, "is_end_of_vally"] == False):  # type: ignore
                    if (
                        vally_df.loc[index - 10, "10均线"]  # type: ignore
                        < vally_df.loc[index - 20, "10均线"]  # type: ignore
                    ) & (
                        vally_df.loc[index - 10, "10均线"] < vally_df.loc[index, "10均线"]  # type: ignore
                    ):
                        vally_df.loc[index, "is_end_of_vally"] = True  # type: ignore
                    else:
                        vally_df.loc[index, "is_end_of_vally"] = False  # type: ignore

            # 重新设置index为日期
            vally_df.set_index("日期", inplace=True)
            # 保留is_end_of_vally=True的行
            vally_df = vally_df[vally_df["is_end_of_vally"] == True]

            # 求出vally_df和ratio_df的日期交集
            intersect_index = vally_df.index.intersection(sliced_ratio_df.index)  # 交集
            # 将df_sma_judge["period"]日期和以上交集相同的日期的行值设为-1
            df_sma_judge[period].loc[intersect_index] = -1

        # 判断存不存在1在df_sma_judge中
        for _, row in df_sma_judge.iterrows():
            if -1 in row.values:
                # 存在1，返回True
                print(row)


class StockAnalyst(BollAnalyst, SMAAnalyst):
    """
    传入股票K线数据，分析后返回买入卖出建议
    """

    def __init__(
        self, stock_code: str, today_date: dt.date, time_window_5: int
    ) -> None:
        self.product_code = stock_code
        self.today_date = today_date
        self.time_window_5 = time_window_5
        self.product_type = "stock"

    def analyze(self):
        # 调用BollAnalyst类的analyze方法
        BollAnalyst.analyze(self)
        # 调用SMAAnalyst类的analyze方法
        SMAAnalyst.analyze(self)
