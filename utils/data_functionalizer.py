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
from utils.enumeration_label import ProductType
from typing import Optional


class DataFunctionalizer:
    def __init__(
        self,
        product_code: str,
        today_date: dt.date | None,
        product_type: ProductType,
        df_product_dict: dict[str, DataFrame] | None,
    ):
        self.product_code = product_code
        self.today_date = today_date or dt.date.today()
        self.product_type = product_type
        self.df_product_dict = df_product_dict or dsp.dataPicker.product_source_picker(
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

    @staticmethod
    def smoother_series(
        data_series: Series,
        sigma: float,  # config
    ) -> Series:
        """
        平滑数据，sigma：越大去噪效果越强，细节越少
        """

        # 检查
        DataFunctionalizer._checking(
            the_type=float,
            data_series=data_series,
            call_name=DataFunctionalizer.smoother_series.__name__,
        )

        # 转换为 NumPy 数组
        data_array = data_series.values
        # 使用高斯滤波器平滑数据
        smoothed_array = gaussian_filter(data_array, sigma=sigma)
        # 将结果转换回 Series
        return_series = Series(smoothed_array, index=data_series.index)

        # 返回结果
        return return_series

    @staticmethod
    def inflection_point_series(data_series: Series, sigma: float) -> Series:  # config
        """
        寻找拐点\n
        输入的series数据应该为浮点型\n
        返回的series数据为浮点型，为拐点对应的值\n
        """

        # 检查
        DataFunctionalizer._checking(
            the_type=float,
            data_series=data_series,
            call_name=DataFunctionalizer.inflection_point_series.__name__,
        )

        # 转换为 NumPy 数组
        data_array = data_series.values
        # 使用高斯滤波器平滑数据
        smoothed_array = gaussian_filter(data_array, sigma=sigma)
        # 将结果转换回 Series
        smoothed_series = Series(smoothed_array, index=data_series.index, name="平滑数据")

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

        # # 绘图准备（点图）
        # smoothed_series.plot(label="Smoothed Data")
        # plt.scatter(
        #     return_series.index,
        #     Series(return_series.values),
        #     label="Inflection Point",
        #     color="red",
        #     alpha=0.3,
        # )
        # # 绘制
        # try:
        #     data_series.plot(label="Raw Data", alpha=0.5)
        #     plt.legend()
        #     plt.show()
        # except TypeError:
        #     plt.legend()
        #     plt.show()

        # 返回结果
        return return_series

    @staticmethod
    def refine_seires(data_series: Series) -> Series:
        """
        精炼数据，将临近数据合并\n
        输入的series数据应该为bool类型，True为临近数据，False为非临近数据\n
        返回的series数据为bool类型，True为精炼数据点，False为普通数据点\n
        精炼的本质是使用特定的卷积核进行卷积，卷积核由积分窗口决定\n
        """

        # 检查
        DataFunctionalizer._checking(
            the_type=bool,
            data_series=data_series,
            call_name=DataFunctionalizer.refine_seires.__name__,
        )

        # 积分窗口
        accumulation_window = int(len(data_series) * 0.005)
        # 卷积核：[1,2,3...,accumulation_window//2,...3,2,1]
        convolution_kernel = []
        for i in range(1, accumulation_window // 2):
            convolution_kernel.append(i)
        for i in range(accumulation_window // 2, 0, -1):
            convolution_kernel.append(i)

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

        # 对所有粘连数据的进行处理：这一步本质上是在计算卷积
        for gdi in glue_df_index:
            for i in range(len(convolution_kernel)):
                # 判断积分值的下标是否在积分表格的范围内
                if (gdi - len(convolution_kernel) // 2 + i >= 0) and (
                    gdi - len(convolution_kernel) // 2 + i < len(data_series)
                ):
                    accumulation_df.loc[
                        gdi - len(convolution_kernel) // 2 + i, "value"
                    ] += convolution_kernel[i]

        # 将积分值转换为Series
        accumulation_series = Series(
            accumulation_df["value"].values, index=accumulation_df["日期"]
        )

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
        height = sum(convolution_kernel)
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

        # # 绘制
        # accumulation_series.plot(label="Accumulation", alpha=0.5)
        # try:
        #     data_series.plot(label="Raw Data", alpha=0.5)
        #     plt.legend()
        #     plt.show()
        # except TypeError:
        #     plt.legend()
        #     plt.show()

        return return_series

    @staticmethod
    def shearing_and_recover(
        data_series: Series, coeff: Optional[np.ndarray] = None
    ) -> tuple[Series, np.ndarray] | Series:
        """
        将数据拟合到x轴上，本质上是进行平移和剪切的线性变换。

        参数：
        - data_series (pd.Series): 输入的数据序列，应为浮点数。
        - coeff (Optional[np.ndarray], optional): 逆变换的依据，一组函数系数。如果为None，将进行线性变换的逆变换。

        返回值：
        - 如果 coeff 为 None，返回一个元组，包含两个值：
            1. 拟合到 x 轴上后的数据序列，为浮点数。
            2. 逆变换的依据，一组函数系数（空数组）。
        - 如果 coeff 不为 None，返回一个序列，为还原后的数据，为浮点数。
        """

        # 检查
        DataFunctionalizer._checking(
            the_type=float,
            data_series=data_series,
            call_name=DataFunctionalizer.shearing_and_recover.__name__,
        )

        # 转换为 NumPy 数组
        data_array = np.array(data_series.values)
        try:
            # 将日期转化为数字，并变为np数组
            num_dates = np.array([mdates.date2num(date) for date in data_series.index])
        except TypeError:
            raise TypeError("日期数据类型错误，无法转换为matplotlib的日期格式")

        # 拟合函数并记录系数
        if coeff is None:
            coefficients = np.polyfit(x=num_dates, y=data_array, deg=5)  # config
        else:
            coefficients = coeff
        fitted_value = np.polyval(coefficients, num_dates)
        # 取拟合曲线的最低点
        minimum_index = np.argmin(fitted_value)
        # print(minimum_index)

        # 拟合函数最低点纵坐标
        minimum_y = fitted_value[minimum_index]
        # 平移数据
        shifted_fitted_value = (
            fitted_value - minimum_y if coeff is None else fitted_value + minimum_y
        )
        shifted_data_array = (
            data_array - minimum_y if coeff is None else data_array + minimum_y
        )
        # 用shifted_data_array和num_dates表示数据点
        points = np.column_stack((num_dates, shifted_data_array))

        # 计算剪切因子k
        k_list = []
        for x in num_dates:
            # np.where返回的是一个元组，所以要加[0]
            y1 = (
                0
                if coeff is None
                else shifted_fitted_value[np.where(num_dates == x)][0]
            )
            x2 = x
            y2 = (
                shifted_fitted_value[np.where(num_dates == x)][0]
                if coeff is None
                else 0
            )
            # 计算剪切因子k
            k = (y1 - y2) / x2
            k_list.append(k)

        # 剪切变换：数据点逐一和对应的剪切矩阵相乘
        # 创建一个空的形状为 (N, 2) 的数组
        new_points_list = []
        for index, k in enumerate(k_list):
            transform_matrix = np.array([[1, 0], [k, 1]])
            new_point = np.matmul(transform_matrix, points[index])
            # sheared_point = points[index]
            new_points_list.append(new_point)

        # 将列表转换为数组
        new_points = np.array(new_points_list)

        # plt.plot(
        #     data_series.index, new_points[:, 1], label="New Data"
        # )  # 所有行第1列，column_stack将点的横坐标和纵坐标分别放在两列，纵向拼接
        # plt.plot(data_series.index, data_array, label="Raw Data")
        # if coeff is None:
        #     plt.plot(
        #         data_series.index,
        #         fitted_value,
        #         label="Raw Linear Fitting",
        #     )
        # plt.legend()
        # plt.show()

        return_series = Series(new_points[:, 1], index=data_series.index)
        return_series = return_series.apply(lambda x: round(x, 4))

        if coeff is None:
            return return_series, coefficients
        else:
            return return_series

    @staticmethod
    def check_cross(
        main_series: Series,
        sub_series: Series,
    ) -> Series:
        """
        检查两条序列的交叉情况\n
        - 参数：\n
            - main_series: 主序列，应为浮点数，index为日期\n
            - sub_series: 副序列，应为浮点数，index为日期\n
        - 返回值：\n
            - 返回一个序列，为交叉点，为日期\n
        """

        # 检查
        for data_series in [main_series, sub_series]:
            DataFunctionalizer._checking(
                the_type=float,
                data_series=data_series,
                call_name=DataFunctionalizer.check_cross.__name__,
            )

        cross_list = []

        # 取较短序列的长度
        length = min(len(main_series), len(sub_series))

        # 检查是否存在交叉点
        for i in range(0, length - 1):
            j = i + 1
            # 上穿
            if (main_series[i] > sub_series[i]) and (main_series[j] <= sub_series[j]):
                cross_list.append(main_series.index[j])
                continue
            # 下穿
            if (main_series[i] < sub_series[i]) and (main_series[j] >= sub_series[j]):
                cross_list.append(main_series.index[j])
                continue

        # 日期从远到近排序
        cross_list.sort()
        # 将列表转换为Series
        return_series = Series(cross_list)
        # print(return_series.head(20))

        return return_series


if __name__ == "__main__":
    test = DataFunctionalizer(
        product_code="002230",
        today_date=dt.date.today(),
        product_type=ProductType.Stock,
        df_product_dict=None,
    )

    # 读取E:\PYTHON\MyPythonProjects\goInvest\data\stock\002230\indicator\002230D_1030_SRLine.csv
    df = pd.read_csv(
        "E:\\PYTHON\\MyPythonProjects\\goInvest\\data\\stock\\002230\\indicator\\002230D_1030_SRLine.csv",
        index_col=0,
    )
    cross = DataFunctionalizer.check_cross(
        df["支撑线"], test.df_product_dict["daily"]["收盘"]
    )
