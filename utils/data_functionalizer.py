"""data_functionalizer.py 试图将数据处理函数化"""

if __name__ == "__main__":
    import sys
    import os

    # 将上级目录加入sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

from pandas import DataFrame, Series
from utils import dataSource_picker as dsp
from scipy.ndimage import gaussian_filter
from utils.enumeration_label import ProductType, SeriesOperation


class DataFunctionalizer:
    def __init__(
        self,
        product_code: str,
        today_date: dt.date | None,
        product_type: ProductType,
        df_product_dict: dict[str, DataFrame] | None,
    ):
        self.product_code = product_code
        self.today_date = today_date if today_date is not None else dt.date.today()
        self.product_type = product_type
        if df_product_dict is not None:
            self.df_product_dict = df_product_dict
        else:
            self.df_product_dict = dsp.dataPicker.product_source_picker(
                product_code=self.product_code,
                today_date=self.today_date,
                product_type=self.product_type,
            )
        self.standard_daily_index = self.df_product_dict["daily"].index
        self.standard_weekly_index = self.df_product_dict["weekly"].index

    @staticmethod
    def _checking(
        data_series: Series, call_name: str, the_type=None, numeric: bool | None = False
    ) -> None:
        """
        检查数据类型是否正确\n
        data_series: 要检查的数据\n
        call_name: 调用的函数名\n
        the_type: 指定的数据类型\n
        numeric: 是否为数值型，默认为False\n
        """
        # 检查数据非空
        if data_series.empty:
            raise ValueError("数据为空，无法进行操作，调用函数为" + call_name)

        # 检查数据类型为可计算类型
        if numeric:
            for index, value in data_series.items():
                # is_numeric_dtype函数的参数应该为Series，所以需要将value转换为Series
                is_numeric = pd.api.types.is_numeric_dtype(Series(value))
                if not is_numeric:
                    raise TypeError(
                        f"{index}的数据类型为{type(value)}，无法进行计算，调用函数为{call_name}"
                    )

        # 检测值的数据类型是否为指定类型
        if the_type is not None:
            for index, value in data_series.items():
                if not isinstance(value, the_type):
                    raise TypeError(
                        f"{index}的数据类型为{type(value)}，应为{the_type}，无法进行操作，调用函数为{call_name}"
                    )

    def series_operator(
        self,
        data_series: Series,
        sigma: float,
        operation: SeriesOperation,
        plot: bool | None = False,
    ) -> Series:
        return_series = Series(index=data_series.index)

        match operation:
            case SeriesOperation.Smoother:
                return_series = DataFunctionalizer.smoother_operation(
                    data_series=data_series, sigma=sigma, plot=plot
                )

            case SeriesOperation.InflectionPoint:
                return_series = DataFunctionalizer.inflection_point_operation(
                    data_series=data_series, sigma=sigma, plot=plot
                )

            case SeriesOperation.Refine:
                return_series = DataFunctionalizer.refine_operation(
                    data_series=data_series, plot=plot
                )

        return return_series

    @staticmethod
    def smoother_operation(
        data_series: Series,
        sigma: float,
        plot: bool | None = False,
    ) -> Series:
        """
        平滑数据，sigma：越大去噪效果越强，细节越少
        """

        # 检查
        DataFunctionalizer._checking(
            the_type=float,
            data_series=data_series,
            call_name=DataFunctionalizer.smoother_operation.__name__,
        )

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
            # 绘制
            try:
                data_series.plot(label="Raw Data", alpha=0.5)
                plt.legend()
                plt.show()
            except TypeError:
                plt.legend()
                plt.show()

        # 返回结果
        return return_series

    @staticmethod
    def inflection_point_operation(
        data_series: Series, sigma: float, plot: bool | None = False
    ) -> Series:
        """
        寻找拐点\n
        输入的series数据应该为浮点型\n
        返回的series数据为浮点型，为拐点对应的值\n
        """

        # 检查
        DataFunctionalizer._checking(
            the_type=float,
            data_series=data_series,
            call_name=DataFunctionalizer.inflection_point_operation.__name__,
        )

        # 转换为 NumPy 数组
        data_array = data_series.values
        # 使用高斯滤波器平滑数据
        smoothed_array = gaussian_filter(data_array, sigma=sigma)
        # 将结果转换回 Series
        smoothed_series = Series(smoothed_array, index=data_series.index, name="平滑数据")
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
        inflecting_series = abs_slope_series[: int(len(abs_slope_series) * 0.05)]
        # 取斜率绝对值小于阈值的值
        threshold = 0.001
        inflecting_series = pd.concat(
            [inflecting_series, abs_slope_series[abs_slope_series < threshold]]
        )
        # 去重
        inflecting_series.drop_duplicates(inplace=True)
        # 取对应下标的值
        return_series = smoothed_series[inflecting_series.index]
        # 排序
        return_series.sort_index(inplace=True)
        # 保留四位小数
        return_series = return_series.apply(lambda x: round(x, 4))
        if plot:
            # 绘图准备（点图）
            plt.scatter(
                return_series.index,
                Series(return_series.values),
                label="Inflection Point",
                color="red",
                alpha=0.3,
            )
            # 绘制
            try:
                data_series.plot(label="Raw Data", alpha=0.5)
                plt.legend()
                plt.show()
            except TypeError:
                plt.legend()
                plt.show()

        # 返回结果
        return return_series

    @staticmethod
    def refine_operation(data_series: Series, plot: bool | None = False) -> Series:
        """
        精炼数据，将粘连数据合并\n
        输入的series数据应该为bool类型，True为粘连数据，False为非粘连数据\n
        返回的series数据为bool类型，True为精炼数据点，False为非精炼数据点\n
        """

        # 检查
        DataFunctionalizer._checking(
            the_type=bool,
            data_series=data_series,
            call_name=DataFunctionalizer.refine_operation.__name__,
        )

        # 积分窗口
        accumulation_window = int(len(data_series) * 0.005)

        # 积分列表
        accumulation_list = []
        for i in range(1, int(accumulation_window / 2)):
            accumulation_list.append(i)
        for i in range(int(accumulation_window / 2), 0, -1):
            accumulation_list.append(i)

        # print(accumulation_list)

        # 创建DataFrame作为积分表格
        accumulation_df = DataFrame(index=data_series.index, columns=["value"])
        # 重设index
        accumulation_df.reset_index(inplace=True)
        # 初始化积分值
        accumulation_df["value"] = 0

        # 取出粘连数据的index
        glue_index = data_series[data_series == True].index
        # 找到accumulation_df中“日期”列和glue_index中的值相同的行的下标
        glue_df_index = accumulation_df[accumulation_df["日期"].isin(glue_index)].index
        # print(glue_df_index)

        # 对所有粘连数据的进行处理
        for gdi in glue_df_index:
            for i in range(len(accumulation_list)):
                # 判断积分值的下标是否在积分表格的范围内
                if (gdi - len(accumulation_list) // 2 + i >= 0) and (
                    gdi - len(accumulation_list) // 2 + i < len(data_series)
                ):
                    accumulation_df.loc[
                        gdi - len(accumulation_list) // 2 + i, "value"
                    ] += accumulation_list[i]

        # 将积分值转换为Series
        accumulation_series = Series(
            accumulation_df["value"].values, index=accumulation_df["日期"]
        )

        if plot:
            # 绘制
            accumulation_series.plot(label="Accumulation", alpha=0.5)

        # 根据积分数据进行精炼，非零积分值从前往后两两相减
        differ_dataframe = DataFrame(
            index=accumulation_series.index, columns=["积分值", "差值", "精炼类型", "精炼值"]
        )
        differ_dataframe["积分值"] = accumulation_series
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
        height = sum(accumulation_list)
        for i in differ_dataframe.index:
            if (i - 1) in differ_dataframe.index:
                # 精炼类型一："tip"
                if (differ_dataframe.loc[i - 1, "差值"] == 1) and (
                    differ_dataframe.loc[i, "差值"] == -1
                ):
                    differ_dataframe.loc[i - 1, "精炼类型"] = "tip"
                    continue

                elif (differ_dataframe.loc[i - 1, "差值"] == 2) and (
                    differ_dataframe.loc[i, "差值"] == 0
                ):
                    differ_dataframe.loc[i - 1, "精炼类型"] = "tip"
                    continue

                # 精炼类型二："platform"
                if (differ_dataframe.loc[i, "积分值"] == height) and (
                    differ_dataframe.loc[i, "差值"] == 1
                ):
                    differ_dataframe.loc[i, "精炼类型"] = "platform"

                elif (differ_dataframe.loc[i, "积分值"] == height) and (
                    differ_dataframe.loc[i, "差值"] == 0
                ):
                    differ_dataframe.loc[i, "精炼类型"] = "platform"

        # print(differ_dataframe)

        # 根据精炼类型判断精炼值
        platform_length = 0
        for i in differ_dataframe.index:
            if differ_dataframe.loc[i, "精炼类型"] == "tip":
                differ_dataframe.loc[i, "精炼值"] = True
            elif differ_dataframe.loc[i, "精炼类型"] == "platform":
                platform_length += 1
            elif differ_dataframe.loc[i, "精炼类型"] == "undefined":
                if platform_length > 0:
                    half_length = platform_length // 2
                    # 当differ_dataframe.loc[i, "精炼类型"]为"undefined"时，才会计算平台中间位置。
                    # 这意味着，平台已经结束，当前的索引i已经超出了平台的范围，所以要额外减一
                    platform_middle = i - half_length - 1
                    differ_dataframe.loc[platform_middle, "精炼值"] = True
                platform_length = 0

        differ_dataframe.to_csv("differ.csv")

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

    # 本函数依赖于standard_daily_index和standard_weekly_index，所以不可能是静态方法
    def _vibration_merge_operation(
        self, data_series: Series, plot: bool | None = False
    ) -> Series:
        """将震荡造成的粘连数据合并"""

        # 检查
        DataFunctionalizer._checking(
            data_series=data_series,
            call_name=DataFunctionalizer._vibration_merge_operation.__name__,
            the_type=None,
            numeric=True,
        )

        # series中的值两两相差
        differ_series = data_series.diff()
        print(f"差值：{differ_series}")

        return Series()

    @staticmethod
    def trend_transform(
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

    # @staticmethod
    # def check_interweave(
    #     main_series: Series,
    #     sub_series: Series,
    #     direction: bool,
    # )->Series:


if __name__ == "__main__":
    test = DataFunctionalizer(
        product_code="002230",
        today_date=dt.date.today(),
        product_type=ProductType.Stock,
        df_product_dict=None,
    )

    # test.data_operator(
    #     data_series=test.df_product_dict["daily"]["收盘"],
    #     sigma=50,
    #     plot=True,
    #     operation=DataOperation.Smoother,
    # )

    inflection_point_series = DataFunctionalizer.inflection_point_operation(
        data_series=test.df_product_dict["daily"]["收盘"], sigma=60, plot=True
    )

    bool_series = Series(index=test.df_product_dict["daily"]["收盘"].index)
    # 初始化bool_series为False
    bool_series = bool_series.apply(lambda x: False)
    # 将拐点的index对应的bool_series值设为True
    bool_series[inflection_point_series.index] = True
    # 处理好的boolSeries传入refine_operation函数处理精炼
    refined_series = DataFunctionalizer.refine_operation(
        data_series=bool_series, plot=True
    )
    # 取出精炼后的拐点
    refined_inflection_point_series = inflection_point_series[refined_series]
    # 绘制
    plt.scatter(
        refined_inflection_point_series.index,
        refined_inflection_point_series,
        label="Refined Inflection Point",
        color="red",
        alpha=0.3,
    )
    test.df_product_dict["daily"]["收盘"].plot(label="Raw Data", alpha=0.5)
    plt.legend()
    plt.show()
    # 消除震荡导致的拐点
    test._vibration_merge_operation(data_series=refined_inflection_point_series)
