"""mySRLine.py"""
if __name__ == "__main__":
    from __init__ import goInvest_path
else:
    from . import goInvest_path

import datetime as dt
import numpy as np
import matplotlib.dates as mdates

from pandas import DataFrame
from utils.data_functionalizer import DataFunctionalizer as dfunc
from utils.myIndicator_abc import MyIndicator
from utils.enumeration_label import ProductType, IndicatorName, DataOperation


class MySRLine(MyIndicator):
    def __init__(
        self,
        data_path: str | None,
        today_date: dt.date | None,
        product_code: str,
        product_type: ProductType,
    ) -> None:
        super().__init__(
            data_path=data_path,
            today_date=today_date,
            product_code=product_code,
            product_type=product_type,
            indicator_name=IndicatorName.SRLine,
        )

    def _remove_redundant_files(self) -> None:
        super()._remove_redundant_files()

    def calculate_indicator(self) -> dict[str, DataFrame]:
        """
        本函数根据时间窗口计算移动平均线
        """
        print(f"正在计算{self.product_code}支撑阻力线...")

        # 清理重复文件
        self._remove_redundant_files()

        # 定义一个字典
        df_srline_dict = {
            "daily": DataFrame(),
            "weekly": DataFrame(),
        }

        # 根据数据和计算支撑/阻力线
        for period in ["daily", "weekly"]:
            closing_price = self.product_df_list[period]["收盘"]
            # 平滑后的收盘价
            closing_price_smoothed = dfunc.data_operator(
                data_series=closing_price, sigma=10, operation=DataOperation.Smoother
            )
            # 将日期转化为数字，并变为np数组
            dates = np.array(closing_price_smoothed.index)
            prices = np.array(closing_price_smoothed.values)

            # 对数据进行线性变换
            transed_data = dfunc._trend_transform(dates, prices, np.array([0, 0]))

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
            original_resistance_line_data = dfunc._trend_transform(
                dates, np.polyval(resistance_line, num_dates), coefficients
            )
            # 支撑线逆变换
            original_support_line_data = dfunc._trend_transform(
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
                file=f"{self.data_path}\\{self.product_code}{period[0].upper()}_{self.today_date.strftime('%m%d')}_SRLine.csv",
                mode="w",
                encoding="utf-8",
            ) as f:
                df_srline_dict[period].to_csv(f, index=True, encoding="utf-8")

        return df_srline_dict


if __name__ == "__main__":
    # 调用函数
    MySRLine(None, dt.date.today(), "002230", ProductType.Stock).calculate_indicator()
