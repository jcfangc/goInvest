if __name__ == "__main__":
    from __init__ import goInvest_path
else:
    from . import goInvest_path

import datetime as dt
import os
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from pandas import DataFrame
from utils import dataSource_picker as dp
from typing import Literal
from utils.data_functionalizer import DataFunctionalizer as dfunc


def trend_transform(
    dates: np.ndarray,
    y_axis: np.ndarray,
    coefficients: np.ndarray,
    plot: bool | None = False,
) -> np.ndarray:
    """
    本函数用于对数据进行线性变换\n
    dates: 当前要处理的x轴（日期）数据\n
    y_axis: 当前要处理的y轴数据\n
    coefficients: \n
    当该参数所代表的一次函数的斜率和截距为0时，表示将原数据进行平移和剪切变换，使其躺倒在x轴上\n
    即先根据截距进行平移变换，然后根据斜率进行剪切变换\n
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
        x, y = np.array([x_axis[-1], np.polyval(transformed_coefficients, x_axis[-1])])
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
            np.matmul(point, anti_shearing_matrix) for point in transformed_data_points
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


def my_support_resistance_line(
    product_code: str,
    today_date: dt.date,
    product_type: Literal["stock"],
) -> dict[str, DataFrame]:
    """
    本函数根据时间窗口计算移动平均线
    """
    print(f"正在计算{product_code}支撑阻力线...")

    # 文件路径
    data_path = f"{goInvest_path}\\data\\kline\\indicator"

    for period_short in ["D", "W"]:
        # 删除过往重复数据
        for file_name in os.listdir(data_path):
            # 防止stock_code和k_period_short为None时参与比较
            if product_code is not None:
                # 指定K线过去的数据会被删除
                if (product_code and period_short and "SRLine") in file_name:
                    # 取得文件绝对路径
                    absfile_path = os.path.join(data_path, file_name)
                    print(f"删除冗余文件\n>>>>{file_name}")
                    # os.remove只能处理绝对路径
                    os.remove(absfile_path)

    # 获取数据
    # 得到包含日K线/周K线的字典
    stock_df = dp.dataPicker.product_source_picker(
        product_code=product_code, today_date=today_date, product_type=product_type
    )

    # 定义一个字典
    df_srline_dict = {
        "daily": DataFrame(),
        "weekly": DataFrame(),
    }

    # 根据数据和计算支撑/阻力线
    for period in ["daily", "weekly"]:
        closing_price = stock_df[period]["收盘"]
        # 平滑后的收盘价
        closing_price_smoothed = dfunc.data_operator(
            data_series=closing_price, sigma=10, operation="smoother"
        )
        # 将日期转化为数字，并变为np数组
        dates = np.array(closing_price_smoothed.index)
        prices = np.array(closing_price_smoothed.values)

        # 对数据进行线性变换
        transed_data = trend_transform(dates, prices, np.array([0, 0]))

        # 将transed_data以第二列为基准大小排序
        sorted_index = np.argsort(transed_data[:, 1])
        sorted_transed_data = transed_data[sorted_index]
        # 拟合点数量
        fit_points_num = int(len(sorted_transed_data) * 0.05)
        # 最大的5%点
        max_points = sorted_transed_data[-fit_points_num:]
        # 最小的5%点
        min_points = sorted_transed_data[:fit_points_num]

        # 拟合阻力线
        resistance_line = np.polyfit(max_points[:, 0], max_points[:, 1], deg=1)
        # 拟合支撑线
        support_line = np.polyfit(min_points[:, 0], min_points[:, 1], deg=1)

        # 支撑线和阻力线进行逆变换
        num_dates = np.array([mdates.date2num(date) for date in dates])
        # 原数据拟合的一次函数参数作为逆变换的参考
        coefficients = np.polyfit(num_dates, prices, deg=1)
        # 阻力线逆变换
        original_resistance_line_data = trend_transform(
            dates, np.polyval(resistance_line, num_dates), coefficients
        )
        # 支撑线逆变换
        original_support_line_data = trend_transform(
            dates, np.polyval(support_line, num_dates), coefficients
        )

        # # 画出拟合的直线和原数据
        # plt.plot(dates, prices, label="Smoothed Data")
        # plt.plot(
        #     dates,
        #     original_resistance_line_data[:, 1],
        #     label="Resistance Line",
        # )
        # plt.plot(
        #     dates,
        #     original_support_line_data[:, 1],
        #     label="Support Line",
        # )
        # plt.legend()
        # plt.show()

        # 创建DataFrame
        df_srline_dict[period] = DataFrame(index=dates, columns=["支撑线", "阻力线"])
        df_srline_dict[period]["支撑线"] = original_support_line_data[:, 1].round(3)
        df_srline_dict[period]["阻力线"] = original_resistance_line_data[:, 1].round(3)
        df_srline_dict[period].index.name = "日期"

        # 检查是否存在nan值
        if df_srline_dict[period].isnull().values.any():
            # 填充nan值
            df_srline_dict[period].fillna(value=0.0, inplace=True)

        # 输出字典到csv文件
        with open(
            file=f"{data_path}\\{product_code}{period[0].upper()}_{today_date.strftime('%m%d')}_SRLine.csv",
            mode="w",
            encoding="utf-8",
        ) as f:
            df_srline_dict[period].to_csv(f, index=True, encoding="utf-8")

    return df_srline_dict


if __name__ == "__main__":
    # 调用函数
    my_support_resistance_line(
        product_code="002230",
        today_date=dt.date.today(),
        product_type="stock",
    )
