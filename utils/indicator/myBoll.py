"""myBoll.py"""

if __name__ == "__main__":
    from __init__ import goInvest_path
else:
    from . import goInvest_path

import datetime as dt


from pandas import DataFrame
from utils.myIndicator_abc import MyIndicator
from utils import dataSource_picker as dp
from utils.enumeration_label import ProductType, IndicatorName


class MyBoll(MyIndicator):
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
            indicator_name=IndicatorName.Boll,
            product_df_dict=product_df_dict,
        )

    def _remove_redundant_files(self) -> None:
        super()._remove_redundant_files()

    def calculate_indicator(self) -> dict[str, DataFrame]:
        """
        本函数计算布林带，返回一个字典，包含不同周期的布林带数据
        """
        print(f"正在计算{self.product_code}的布林带...")

        # 清理重复文件
        self._remove_redundant_files()

        # 定义布林带的时间窗口
        time_window = 10

        # 定义一个字典，用于存放返回的不同周期bollinger数据
        df_bollinger_dict = {
            "daily": DataFrame(),
            "weekly": DataFrame(),
        }

        dict_df_sma = dp.dataPicker.indicator_source_picker(
            product_code=self.product_code,
            today_date=self.today_date,
            indicator_name=IndicatorName.SMA,
            product_type=self.product_type,
            product_df_dict=self.product_df_dict,
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
            k = 2
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
        super().save_indicator(df_dict=df_bollinger_dict)
        # 返回字典
        return df_bollinger_dict

    def analyze(self) -> list[DataFrame]:
        # 获取股票K线boll数据
        dict_boll = super().pre_analyze()
        # 调用策略函数
        sma_mutiline_judge = self._boll_strategy(dict_boll)
        # 返回策略
        return [sma_mutiline_judge]

    def _boll_strategy(
        self,
        dict_boll: dict[str, DataFrame],
    ) -> DataFrame:
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
                        # 3. 上影线小于下轨线，df_boll_judge的值为1
                        if ohlc_data_date["最高"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 1
                        # 2. 收盘价等于开盘价小于下轨线，df_boll_judge的值为0.6
                        elif ohlc_data_date["收盘"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 0.6
                        # 1. 下影线小于下轨线，df_boll_judge的值为0.2
                        elif ohlc_data_date["最低"] <= boll_data_date["下轨"]:
                            df_boll_judge.loc[date, period] = 0.2

        # 保存策略
        super().save_strategy(
            df_judge=df_boll_judge, func_name=MyBoll._boll_strategy.__name__
        )
        # 返回策略
        return df_boll_judge


if __name__ == "__main__":
    # 测试
    MyBoll(
        None, dt.date.today(), "002230", ProductType.Stock, None
    ).calculate_indicator()
