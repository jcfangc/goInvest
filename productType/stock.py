import datetime as dt


from utils import dataSource_picker as dsp
from utils import data_analyst as da
from pandas import Series


class Stock:
    """
    这是一个stock对象，创建该对象请调用构造函数并传入A股股票代码
    """

    def __init__(self, requirement: Series) -> None:
        self.stock_code = requirement["identityCode"]
        self.today_date = dt.datetime.today().date()
        self.time_window_5 = int(5)

    # 获取指定产品的日K/周K
    def obtain_kline(self) -> None:
        # 获取数据
        dsp.dataPicker.product_source_picker(
            self.stock_code,
            today_date=self.today_date,
            product_type="stock",
        )

    # 分析指定产品的日K/周K，生成分析报告
    def analyze_stock(self) -> None:
        # 调用分析函数
        da.StockAnalyst(
            stock_code=self.stock_code,
            today_date=self.today_date,
            time_window_5=self.time_window_5,
        ).analyze()
