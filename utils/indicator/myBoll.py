"""myBoll.py"""
if __name__ == "__main__":
    from __init__ import goInvest_path
else:
    from . import goInvest_path

import datetime as dt

from pandas import DataFrame
from utils import dataSource_picker as dp
from utils.myIndicator_abc import MyIndicator
from utils.enumeration_label import ProductType, IndicatorName


class MyBoll(MyIndicator):
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
            indicator_name=IndicatorName.Boll,
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

        for period in ["daily", "weekly"]:
            # 获取移动平均线中间SMA
            mid_sma_series = dp.dataPicker.indicator_source_picker(
                product_code=self.product_code,
                today_date=self.today_date,
                indicator_name=IndicatorName.SMA,
                product_type=self.product_type,
            )[period][
                f"{time_window}均线"
            ]  # Name: 10均线, dtype: float64, index.name: 日期

            # 数据对齐
            stock_close_series = self.product_df_list[period][
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

            # 检查是否存在nan值
            if df.isnull().values.any():
                # 填充nan值
                df.fillna(value=0.0, inplace=True)

            # 输出字典为csv文件
            with open(
                file=f"{self.data_path}\\{self.product_code}{period[0].upper()}_{self.today_date.strftime('%m%d')}_Boll.csv",
                mode="w",
                encoding="utf-8",
            ) as f:
                df.to_csv(f, index=True, encoding="utf-8")

            match period:
                case "daily":
                    df_bollinger_dict["daily"] = df
                case "weekly":
                    df_bollinger_dict["weekly"] = df

        return df_bollinger_dict


if __name__ == "__main__":
    # 测试
    MyBoll(None, dt.date.today(), "002230", ProductType.Stock).calculate_indicator()
