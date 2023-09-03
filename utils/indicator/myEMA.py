"""myEMA.py"""
if __name__ == "__main__":
    from __init__ import goInvest_path
else:
    from . import goInvest_path

import datetime as dt

from pandas import DataFrame
from utils.myIndicator_abc import MyIndicator
from utils.enumeration_label import ProductType, IndicatorName


class MyEMA(MyIndicator):
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
            indicator_name=IndicatorName.EMA,
        )

    def _remove_redundant_files(self) -> None:
        super()._remove_redundant_files()

    def calculate_indicator(self) -> dict[str, DataFrame]:
        """
        本函数根据时间窗口计算移动平均线
        """
        print(f"正在计算{self.product_code}指数移动平均线...")

        # 清理重复文件
        self._remove_redundant_files()

        # 定义一个字典，用于存放返回的不同周期sma数据
        df_ema_dict = {
            "daily": DataFrame(),
            "weekly": DataFrame(),
        }

        # 根据数据和时间窗口滚动计算SMA
        for period in ["daily", "weekly"]:
            closing_price = self.product_df_list[period]["收盘"]
            # 5时间窗口，数值取三位小数
            ema_5 = closing_price.ewm(span=5, adjust=False).mean()
            # 10时间窗口，数值取三位小数
            ema_10 = closing_price.ewm(span=10, adjust=False).mean()
            # 20时间窗口，数值取三位小数
            ema_20 = closing_price.ewm(span=20, adjust=False).mean()
            # 50时间窗口，数值取三位小数
            ema_50 = closing_price.ewm(span=50, adjust=False).mean()
            # 150时间窗口，数值取三位小数
            ema_150 = closing_price.ewm(span=150, adjust=False).mean()

            # 均线数据汇合
            df_ema_dict[period] = DataFrame(
                {
                    # 日期作为索引
                    # 均线数据
                    "5均线": ema_5.round(3),
                    "10均线": ema_10.round(3),
                    "20均线": ema_20.round(3),
                    "50均线": ema_50.round(3),
                    "150均线": ema_150.round(3),
                }
            )

            # 检查是否存在nan值
            if df_ema_dict[period].isnull().values.any():
                # 填充nan值
                df_ema_dict[period].fillna(value=0.0, inplace=True)

            # 输出字典到csv文件
            with open(
                file=f"{self.data_path}\\{self.product_code}{period[0].upper()}_{self.today_date.strftime('%m%d')}_EMA.csv",
                mode="w",
                encoding="utf-8",
            ) as f:
                df_ema_dict[period].to_csv(f, index=True, encoding="utf-8")

        return df_ema_dict


if __name__ == "__main__":
    # 调用函数stock_EMA
    MyEMA(None, dt.date.today(), "002230", ProductType.Stock).calculate_indicator()
