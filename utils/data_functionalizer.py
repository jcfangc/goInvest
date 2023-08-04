"""data_functionalizer.py 试图将数据处理函数化"""

if __name__ == "__main__":
    from __init__ import goInvest_path
else:
    from . import goInvest_path

import datetime as dt
import os
import pandas as pd
import matplotlib.pyplot as plt


from pandas import DataFrame, Series
from goInvest.utils import dataSource_picker as dsp
from scipy.ndimage import gaussian_filter
from typing import Literal


class DataFunctionalizer:
    def __init__(
        self, product_code: str, today_date: dt.date, product_type: Literal["stock"]
    ):
        self.product_code = product_code
        self.today_date = today_date
        self.product_type = product_type
        self.df_product_dict = dsp.dataPicker.product_source_picker(
            product_code=self.product_code,
            today_date=self.today_date,
            product_type=self.product_type,
        )

    @staticmethod
    def data_operator(
        data_series: Series,
        sigma: float,
        operation: Literal["smoother", "inflectionPoint", "refine"],
        plot: bool | None = False,
    ) -> Series:
        """
        sigma：越大去噪效果越强，细节越少
        """

        return_series = Series(index=data_series.index)

        match operation:
            case "smoother":
                """
                平滑数据
                """
                # 转换为 NumPy 数组
                data_array = data_series.values
                # 使用高斯滤波器平滑数据
                smoothed_array = gaussian_filter(data_array, sigma=sigma)
                # 将结果转换回 Series
                return_series = smoothed_series = Series(
                    smoothed_array, index=data_series.index
                )
                if plot:
                    # 返回平滑后的数据
                    smoothed_series.plot(label="Smoothed Data")
            case "inflectionPoint":
                """
                寻找拐点
                """
                # 转换为 NumPy 数组
                data_array = data_series.values
                # 使用高斯滤波器平滑数据
                smoothed_array = gaussian_filter(data_array, sigma=sigma)
                # 将结果转换回 Series
                smoothed_series = Series(smoothed_array, index=data_series.index)
                if plot:
                    smoothed_series.plot(label="Smoothed Data")
                # 滚动计算平滑后两点之间的斜率
                slope_series = Series(index=data_series.index)
                for i in range(1, len(smoothed_array)):
                    # 计算斜率
                    slope = smoothed_array[i] - smoothed_array[i - 1]
                    # 将斜率存入series
                    slope_series[i - 1] = slope
                # 关注斜率的绝对值
                abs_slope_series = slope_series.abs()
                # 根据绝对值重新排序
                abs_slope_series.sort_values(inplace=True, ascending=True)
                # 取斜率绝对值的最小值的5%
                inflecting_series = abs_slope_series[
                    : int(len(abs_slope_series) * 0.05)
                ]
                # 取斜率绝对值小于阈值的值
                threshold = 0.001
                inflecting_series = pd.concat(
                    [inflecting_series, abs_slope_series[abs_slope_series < threshold]]
                )
                # 去重
                inflecting_series.drop_duplicates(inplace=True)
                # inflecting_series中的数据顺序两两相减，
                # 取对应下标的值
                return_series = smoothed_series[inflecting_series.index]
                if plot:
                    # 绘图准备（点图）
                    plt.scatter(
                        return_series.index,
                        Series(return_series.values),
                        label="Inflection Point",
                        color="red",
                        alpha=0.3,
                    )

                # 构造布尔型series，用于判断是否为拐点
                bool_series = Series(index=data_series.index)
                # 初始化bool_series为False
                bool_series = bool_series.apply(lambda x: False)
                # 将拐点的index对应的bool_series值设为True
                bool_series[return_series.index] = True
                # 递归调用函数进行精炼
                refined_bool_series = DataFunctionalizer.data_operator(
                    bool_series, sigma, "refine"
                )
                # 取出refined_bool_series中为True的index
                refined_index = refined_bool_series[refined_bool_series == True].index
                # 最终的return_series
                return_series = smoothed_series[refined_index]

                if plot:
                    # 绘图准备（点图）
                    plt.scatter(
                        return_series.index,
                        Series(return_series.values),
                        label="Refined points",
                        color="green",
                    )

            case "refine":
                """
                输入的series数据应该为bool类型，True为粘连数据，False为非粘连数据
                精炼数据，将粘连数据合并
                """
                # 检测值的数据类型是否为布尔型
                if data_series.dtype != bool:
                    raise TypeError("输入的series元素数据类型应为布尔型")

                # 积分窗口
                accumulation_window = int(len(data_series) * 0.005)

                # 积分列表
                accumulation_list = []
                for i in range(1, int(accumulation_window / 2)):
                    accumulation_list.append(i)
                for i in range(int(accumulation_window / 2), 0, -1):
                    accumulation_list.append(i)

                # 创建DataFrame作为积分表格
                accumulation_df = DataFrame(index=data_series.index, columns=["value"])
                # 重设index
                accumulation_df.reset_index(inplace=True)
                # 初始化积分值
                accumulation_df["value"] = 0

                # 取出粘连数据的index
                glue_index = data_series[data_series == True].index
                # 找到accumulation_df中“日期”列和glue_index中的值相同的行的下标
                glue_df_index = accumulation_df[
                    accumulation_df["日期"].isin(glue_index)
                ].index

                # 计算偏移值（是中间元素的偏左或偏右多少位）
                offset = -int((len(accumulation_list) - 1) / 2)
                # 遍历这些下标
                for marks in accumulation_list:
                    for i in glue_df_index:
                        if (i + offset) in accumulation_df.index:
                            accumulation_df.loc[i + offset, "value"] += marks
                    offset += 1

                # 将积分值转换为Series
                accumulation_series = Series(
                    accumulation_df["value"].values, index=accumulation_df["日期"]
                )

                if plot:
                    # 绘制
                    accumulation_series.plot(label="Accumulation", alpha=0.5)

                # 根据积分数据进行精炼，非零积分值从前往后两两相减
                differ_dataframe = DataFrame(
                    index=accumulation_series.index, columns=["差值", "精炼类型", "精炼值"]
                )
                # 重设index
                differ_dataframe.reset_index(inplace=True)
                # 初始化
                differ_dataframe["差值"] = 0
                differ_dataframe["精炼类型"] = "undefined"
                differ_dataframe["精炼值"] = False
                # 计算差值
                for i in range(1, len(accumulation_series)):
                    if accumulation_series[i] != 0:
                        differ_dataframe.loc[i, "差值"] = (
                            accumulation_series[i] - accumulation_series[i - 1]
                        )
                # 根据差值判断精炼类型
                for i in differ_dataframe.index:
                    if (i - 1) in differ_dataframe.index:
                        # 精炼类型一："tip"
                        if (differ_dataframe.loc[i - 1, "差值"] < 0) and (differ_dataframe.loc[i, "差值"] > 0):  # type: ignore
                            differ_dataframe.loc[i, "精炼类型"] = "tip"
                        # 精炼类型二："platform"
                        platform_stage = False
                        if (differ_dataframe.loc[i - 1, "差值"] < 0) and (differ_dataframe.loc[i, "差值"] == 0):  # type: ignore
                            differ_dataframe.loc[i, "精炼类型"] = "platform"
                            platform_stage = True
                        elif (
                            (differ_dataframe.loc[i - 1, "差值"] == 0)
                            and (differ_dataframe.loc[i, "差值"] == 0)
                            and platform_stage  # 确保是在平台上
                        ):
                            differ_dataframe.loc[i, "精炼类型"] = "platform"
                        elif (differ_dataframe.loc[i - 1, "差值"] == 0) and (differ_dataframe.loc[i, "差值"] > 0):  # type: ignore
                            differ_dataframe.loc[i, "精炼类型"] = "platform"
                            platform_stage = False

                # 根据精炼类型判断精炼值
                platform_length = 0
                for i in differ_dataframe.index:
                    if differ_dataframe.loc[i, "精炼类型"] == "tip":
                        differ_dataframe.loc[i, "精炼值"] = True
                    elif differ_dataframe.loc[i, "精炼类型"] == "platform":
                        platform_length += 1
                    elif differ_dataframe.loc[i, "精炼类型"] == "undefined":
                        if platform_length > 0:
                            differ_dataframe.loc[
                                i - int(platform_length / 2), "精炼值"
                            ] = True
                        platform_length = 0

                # 恢复index
                differ_dataframe.set_index("日期", inplace=True)
                # 将精炼值转换为Series
                return_series = differ_dataframe["精炼值"]

        if plot:
            # 绘制
            try:
                data_series.plot(label="Raw Data", alpha=0.5)
                plt.legend()
                plt.show()
            except TypeError:
                plt.legend()
                plt.show()

        return return_series


if __name__ == "__main__":
    test = DataFunctionalizer(
        product_code="002230", today_date=dt.date.today(), product_type="stock"
    )

    # test.data_operator(
    #     data_series=test.df_product_dict["daily"]["收盘"],
    #     sigma=10,
    #     plot=True,
    #     operation="smoother",
    # )

    test_series = test.data_operator(
        data_series=test.df_product_dict["daily"]["收盘"],
        sigma=50,
        plot=False,
        operation="inflectionPoint",
    )

    bool_series = Series(index=test.df_product_dict["daily"]["收盘"].index)
    # 初始化bool_series为False
    bool_series = bool_series.apply(lambda x: False)
    # 将拐点的index对应的bool_series值设为True
    bool_series[test_series.index] = True

    test.data_operator(
        data_series=bool_series,
        sigma=50,
        plot=True,
        operation="refine",
    )
