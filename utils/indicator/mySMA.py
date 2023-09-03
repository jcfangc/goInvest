"""mySMA.py"""
if __name__ == "__main__":
    from __init__ import goInvest_path
else:
    from . import goInvest_path

import datetime as dt

from pandas import DataFrame
from utils.myIndicator_abc import MyIndicator
from utils.enumeration_label import ProductType, IndicatorName


class MySMA(MyIndicator):
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
            indicator_name=IndicatorName.SMA,
        )

    def _remove_redundant_files(self) -> None:
        super()._remove_redundant_files()

    def calculate_indicator(self) -> dict[str, DataFrame]:
        """
        本函数根据时间窗口计算移动平均线
        """
        print(f"正在计算{self.product_code}移动平均线...")

        # 清理重复文件
        self._remove_redundant_files()

        # 定义一个字典，用于存放返回的不同周期sma数据
        df_sma_dict = {
            "daily": DataFrame(),
            "weekly": DataFrame(),
        }

        print(self.product_df_list["daily"].head())

        # 根据数据和时间窗口滚动计算SMA
        for period in ["daily", "weekly"]:
            closing_price = self.product_df_list[period]["收盘"]
            # 5时间窗口，数值取三位小数
            sma_5 = closing_price.rolling(5).mean()
            # 10时间窗口，数值取三位小数
            sma_10 = closing_price.rolling(10).mean()
            # 20时间窗口，数值取三位小数
            sma_20 = closing_price.rolling(20).mean()
            # 50时间窗口，数值取三位小数
            sma_50 = closing_price.rolling(50).mean()
            # 150时间窗口，数值取三位小数
            sma_150 = closing_price.rolling(150).mean()

            # 均线数据汇合
            df_sma_dict[period] = DataFrame(
                {
                    # 日期作为索引
                    # 均线数据
                    "5均线": sma_5.round(3),
                    "10均线": sma_10.round(3),
                    "20均线": sma_20.round(3),
                    "50均线": sma_50.round(3),
                    "150均线": sma_150.round(3),
                }
            )

            # 检查是否存在nan值
            if df_sma_dict[period].isnull().values.any():
                # 填充nan值
                df_sma_dict[period].fillna(value=0.0, inplace=True)

            # 输出字典到csv文件
            with open(
                file=f"{self.data_path}\\{self.product_code}{period[0].upper()}_{self.today_date.strftime('%m%d')}_SMA.csv",
                mode="w",
                encoding="utf-8",
            ) as f:
                df_sma_dict[period].to_csv(f, index=True, encoding="utf-8")

        return df_sma_dict


if __name__ == "__main__":
    MySMA(None, dt.date.today(), "002230", ProductType.Stock).calculate_indicator()
