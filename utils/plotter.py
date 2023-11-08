if __name__ == "__main__":
    import sys
    import os

    # 将上级目录加入sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

from pandas import DataFrame
from utils import dataSource_picker as dp
from utils.enumeration_label import ProductType, IndicatorName, StrategyName
from config import __BASE_PATH__, do_logging
from matplotlib import pyplot as plt
from typing import Optional
from matplotlib.figure import Figure
import datetime as dt


logger = do_logging()
# 添加字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
# 设置全局样式
plt.style.use("dark_background")
# 通用参数
params = {}


class Plotter:
    """一个绘制数据的类"""

    def __init__(
        self,
        data_path: Optional[str],
        today_date: Optional[dt.date],
        product_code: str,
        product_type: ProductType,
        product_df_dict: Optional[dict[str, DataFrame]],
    ) -> None:
        self.data_path = (
            data_path or f"{__BASE_PATH__}\\data\\{product_type.value}\\{product_code}"
        )
        self.today_date = today_date or dt.date.today()
        self.product_code = product_code
        self.product_type = product_type
        self.product_df_dict = product_df_dict or dp.dataPicker.product_source_picker(
            product_code=product_code,
            today_date=today_date,
            product_type=product_type,
        )
        params["today_date"] = today_date
        params["product_code"] = product_code
        params["product_type"] = product_type
        params["product_df_dict"] = product_df_dict

    def plot_all(
        self,
        indicator_name_for_strategy: IndicatorName,
        strategy_name: StrategyName,
        indicator_names: list[IndicatorName],
        period: str = "daily",
        save: bool = True,
        show: bool = False,
        from_date: int = 200,
        to_date: int = 1,
    ):
        """绘制所有图"""
        self.plot_kline(
            period=period, save=save, show=show, from_date=from_date, to_date=to_date
        )
        self.plot_indicator(
            indicator_names=indicator_names,
            period=period,
            save=save,
            show=show,
            from_date=from_date,
            to_date=to_date,
        )
        self.plot_strategy(
            indicator_name=indicator_name_for_strategy,
            strategy_name=strategy_name,
            period=period,
            save=save,
            show=show,
            from_date=from_date,
            to_date=to_date,
        )

    def plot_kline(
        self,
        period: str = "daily",
        save: bool = True,
        show: bool = False,
        from_date: int = 200,
        to_date: int = 1,
    ) -> Figure:
        """
        绘制K线图
        - 参数：
            - period：K线周期，可选值为daily、weekly
            - save：是否保存图片，默认为True，保存图片
            - show：是否显示图片，默认为False，不显示图片
            - from_date：从倒数第from_date天开始绘制，默认为200
            - to_date：到倒数第to_date天结束绘制，默认为1
        """
        # 读取数据，取倒数from_date到倒数to_date的数据
        data_df = self.product_df_dict[period][-from_date:-to_date]
        # 绘制蜡烛图
        fig, ax = plt.subplots(figsize=(20, 10))
        # 绘制k线图，黑底白线
        ax.plot(data_df.index, data_df["收盘"], color="white")
        # 设置标题
        plt.title(f"{self.product_code} {period}", color="white")
        # 保存图片
        if save:
            plt.savefig(f"{self.data_path}\\plot\\{self.product_code}{period}.svg")
        # 显示图片
        if show:
            plt.show()

        return fig

    def plot_indicator(
        self,
        indicator_names: list[IndicatorName],
        period: str = "daily",
        save: bool = True,
        show: bool = False,
        from_date: int = 200,
        to_date: int = 1,
    ) -> list[Figure]:
        """
        绘制指标图

        - 参数：
            - indicator_names：指标名称列表
            - period：K线周期，可选值为daily、weekly
            - save：是否保存图片，默认为True，保存图片
            - show：是否显示图片，默认为False，不显示图片
        """
        figs = []
        for indicator_name in indicator_names:
            fig, ax = plt.subplots(figsize=(20, 10))
            # 读取数据
            data_df = dp.dataPicker.indicator_source_picker(
                indicator_name=indicator_name, **params
            )[period][-from_date:-to_date]
            # 获取数据序号和列名
            index_list = data_df.index
            column_list = data_df.columns
            # 绘制指标图
            for column in column_list:
                ax.plot(index_list, data_df[column], label=column, linewidth=0.25)
            # 设置标题
            plt.title(f"{self.product_code}_{period}_{indicator_name.value}")
            # 设置图例
            plt.legend()
            figs.append(fig)
            # 保存图片
            if save:
                plt.savefig(
                    f"{self.data_path}\\plot\\{self.product_code}_{period}_{indicator_name.value}.svg"
                )
            # 显示图片
            if show:
                plt.show()

        return figs

    def plot_strategy(
        self,
        indicator_name: IndicatorName,
        strategy_name: StrategyName,
        period: str = "daily",
        save: bool = True,
        show: bool = False,
        from_date: int = 200,
        to_date: int = 1,
    ) -> Figure:
        """绘制策略图"""

        fig, ax = plt.subplots(figsize=(20, 10))
        # 读取数据
        strategy_values = dp.dataPicker.strategy_source_picker(
            strategy_name=strategy_name, **params
        )[period][-from_date:-to_date]
        # 读取数据，取倒数from_date到倒数to_date的数据
        data_df = self.product_df_dict[period][-from_date:-to_date]
        # 读取技术指标数据
        indicator_df = dp.dataPicker.indicator_source_picker(
            indicator_name=indicator_name, **params
        )[period][-from_date:-to_date]
        # 绘制指标图
        for column in indicator_df.columns:
            ax.plot(
                indicator_df.index, indicator_df[column], label=column, linewidth=0.25
            )
        # 绘制k线图
        ax.plot(data_df.index, data_df["收盘"], color="white", linewidth=0.25)
        # 根据策略数值绘制红点（卖出操作）和绿点（买入操作）
        for i, value in enumerate(strategy_values):
            if value > 0:
                # 买入操作，绘制绿点
                ax.scatter(
                    data_df.index[i],
                    data_df["收盘"].iloc[i],
                    color="green",
                    alpha=value,
                    s=5,
                )
            elif value < 0:
                # 卖出操作，绘制红点
                ax.scatter(
                    data_df.index[i],
                    data_df["收盘"].iloc[i],
                    color="red",
                    alpha=abs(value),
                    s=5,
                )

        # 设置图例
        plt.legend()
        # 设置标题
        plt.title(f"{self.product_code}_{period}{strategy_name.value}", color="white")
        # 保存图片
        if save:
            plt.savefig(
                f"{self.data_path}\\plot\\{self.product_code}_{period}_{strategy_name.value}.svg"
            )
        # 显示图片
        if show:
            plt.show()

        return fig


if __name__ == "__main__":
    Plotter(
        data_path=None,
        today_date=dt.date.today(),
        product_code="600418",
        product_type=ProductType.Stock,
        product_df_dict=None,
    ).plot_strategy(
        indicator_name=IndicatorName.SRLine,
        strategy_name=StrategyName.SRLine_PSS,
        from_date=2000,
        to_date=1,
    )
