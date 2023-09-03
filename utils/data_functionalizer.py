"""data_functionalizer.py 试图将数据处理函数化"""

if __name__ == "__main__":
    from __init__ import goInvest_path
else:
    from . import goInvest_path

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

from pandas import DataFrame, Series
from goInvest.utils import dataSource_picker as dsp
from scipy.ndimage import gaussian_filter
from utils.enumeration_label import ProductType, DataOperation


class DataFunctionalizer:
    def __init__(
        self, product_code: str, today_date: dt.date, product_type: ProductType
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
        operation: DataOperation,
        plot: bool | None = False,
    ) -> Series:
        """
        sigma：越大去噪效果越强，细节越少
        """

        return_series = Series(index=data_series.index)

        match operation:
            case DataOperation.Smoother:
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
            case DataOperation.InflectionPoint:
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

                # # 以下注释掉的代码是在测试对拐点结果的精炼效果
                # # 构造布尔型series，用于判断是否为拐点
                # bool_series = Series(index=data_series.index)
                # # 初始化bool_series为False
                # bool_series = bool_series.apply(lambda x: False)
                # # 将拐点的index对应的bool_series值设为True
                # bool_series[return_series.index] = True
                # # 递归调用函数进行精炼
                # refined_bool_series = DataFunctionalizer.data_operator(
                #     bool_series, sigma, DataOperation.Refine, plot=True
                # )
                # # 取出refined_bool_series中为True的index
                # refined_index = refined_bool_series[refined_bool_series == True].index
                # # 最终的return_series
                # return_series = smoothed_series[refined_index]

                # if plot:
                #     # 绘图准备（点图）
                #     plt.scatter(
                #         return_series.index,
                #         Series(return_series.values),
                #         label="Refined points",
                #         color="green",
                #     )

            case DataOperation.Refine:
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

                print(accumulation_list)

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
                # print(glue_df_index)

                # 计算偏移值（是中间元素的偏左或偏右多少位）
                offset = -int(len(accumulation_list))
                # 遍历这些下标，添加积分
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
                        elif (differ_dataframe.loc[i - 1, "差值"] == 0) and (differ_dataframe.loc[i, "差值"] > 0) and platform_stage:  # type: ignore
                            differ_dataframe.loc[i, "精炼类型"] = "platform"
                            platform_stage = False

                print(differ_dataframe)

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
                                i - int(platform_length * 1 / 2), "精炼值"
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

    @staticmethod
    def _trend_transform(
        dates: np.ndarray,
        y_axis: np.ndarray,
        coefficients: np.ndarray,
        plot: bool | None = False,
    ) -> np.ndarray:
        """
        如果需要将k线数据的趋势影响消除，可以尝试使用本函数用于对数据进行线性变换\n
        dates: 当前要处理的x轴（日期）数据\n
        y_axis: 当前要处理的y轴数据\n
        coefficients: \n
        当该参数所代表的一次函数的斜率和截距为0时，表示将原数据进行平移和剪切变换，使其躺倒在x轴上\n
        即先根据截距进行平移变换使得一次函数经过原点，然后根据斜率进行剪切变换使得一次函数躺倒在x轴\n
        当该参数做代表的一次函数的斜率和截距为非零时，表示将当前数据根据斜率和截距进行变换\n
        即先根据斜率进行剪切变换，然后根据截距进行平移变换\n
        plot: 是否绘制图形，默认不绘制\n
        返回值: 变换后的x和y轴数据
        """
        try:
            # 生成x轴数据
            x_axis = np.array([mdates.date2num(date) for date in dates])
        except TypeError:
            raise TypeError("日期数据类型错误，无法转换为matplotlib的日期格式")

        # 斜率
        slope = coefficients[0]
        # 截距
        intercept = coefficients[1]

        if slope == 0 and intercept == 0:
            # 获得数据对应的一次函数参数
            data_coefficients = np.polyfit(x_axis, y_axis, deg=1)
            data_slope = data_coefficients[0]
            data_intercept = data_coefficients[1]

            # 平移变换
            transformed_pairs = np.column_stack((x_axis, y_axis))
            transformed_pairs[:, 1] = transformed_pairs[:, 1] - data_intercept
            # 平移后的一次函数参数，经实验截距为0
            transformed_coefficients = [data_slope, 0]

            # 剪切变换
            # 取原一次函数的最后一个坐标点，计算剪切因子k
            x, y = np.array(
                [x_axis[-1], np.polyval(transformed_coefficients, x_axis[-1])]
            )
            # 计算剪切因子k
            try:
                k = (-y) / x
            except ZeroDivisionError:
                raise ZeroDivisionError("最后一点是原点数据，无法计算剪切因子k")
            # 创建剪切矩阵
            shearing_matrix = np.array([[1, k], [0, 1]])

            # 数据进行线性变换
            transformed_data_points = np.array(
                [np.matmul(point, shearing_matrix) for point in transformed_pairs]
            )

            # 画出拟合的直线和数据
            if plot:
                plt.plot(dates, y_axis, label="Smoothed Data")
                plt.plot(
                    dates,
                    data_slope * x_axis + data_intercept,
                    label="Trend Line",
                )
                plt.plot(dates, transformed_data_points[:, 1], label="Transformed Data")
                # 画出y=0的直线
                plt.plot(dates, np.zeros(len(dates)), label="Transformed Trend Line")
                plt.legend()
                plt.show()

            # x轴数据转换为日期
            # transformed_data_points[:, 0] = np.array(dates)

            return transformed_data_points

        # 逆变换
        else:
            original_coefficients = coefficients
            transformed_data_points = np.column_stack((x_axis, y_axis))
            # 剪切变换
            temp_coefficients = [original_coefficients[0], 0]
            # 取原一次函数的最后一个坐标点，计算剪切因子k
            x, y = np.array([x_axis[-1], np.polyval(temp_coefficients, x_axis[-1])])
            # 计算剪切因子k
            try:
                k = (-y) / x
            except ZeroDivisionError:
                raise ZeroDivisionError("最后一点是原点数据，无法计算剪切因子k")
            # 创建剪切矩阵
            anti_shearing_matrix = np.array([[1, -k], [0, 1]])
            original_data_points = [
                np.matmul(point, anti_shearing_matrix)
                for point in transformed_data_points
            ]
            original_data_points = np.array(original_data_points)

            # 平移变换
            original_data_points[:, 1] = original_data_points[:, 1] + intercept

            # 画出拟合的直线和数据
            if plot:
                plt.plot(dates, y_axis, label="Transformed Data")
                plt.plot(dates, np.zeros(len(x_axis)), label="Transformed Trend Line")
                plt.plot(dates, original_data_points[:, 1], label="Original Data")
                plt.plot(
                    dates,
                    np.polyval(original_coefficients, x_axis),
                    label="Original Trend Line",
                )
                plt.legend()
                plt.show()

            # x轴数据转换为日期
            # original_data_points[:, 0] = np.array(dates)

            return original_data_points


if __name__ == "__main__":
    test = DataFunctionalizer(
        product_code="002230",
        today_date=dt.date.today(),
        product_type=ProductType.Stock,
    )

    # test.data_operator(
    #     data_series=test.df_product_dict["daily"]["收盘"],
    #     sigma=50,
    #     plot=True,
    #     operation=DataOperation.Smoother,
    # )

    test_series = test.data_operator(
        data_series=test.df_product_dict["daily"]["收盘"],
        sigma=60,
        plot=False,
        operation=DataOperation.InflectionPoint,
    )

    bool_series = Series(index=test.df_product_dict["daily"]["收盘"].index)
    # 初始化bool_series为False
    bool_series = bool_series.apply(lambda x: False)
    # 将拐点的index对应的bool_series值设为True
    bool_series[test_series.index] = True

    test.data_operator(
        data_series=bool_series,
        sigma=60,
        plot=True,
        operation=DataOperation.Refine,
    )
