"""enumeration_label.py 用于创建一些通用的枚举类"""
from enum import Enum


class IndicatorName(Enum):
    Boll = "Boll"
    EMA = "EMA"
    RSI = "RSI"
    SMA = "SMA"
    SRLine = "SRLine"


class ProductType(Enum):
    Stock = "stock"


class StrategyName(Enum):
    # from indicator.myBoll import MyBoll
    # from indicator.myEMA import MyEMA
    # from indicator.mySMA import MySMA
    # from indicator.mySRLine import MySRLine

    # Boll_BLL = MyBoll._boll_strategy.__name__
    # SMA_MTL = MySMA._mutiline_strategy.__name__
    # EMA_MTL = MyEMA._mutiline_strategy.__name__
    # SRLine_PSS = MySRLine._pressure_area_strategy.__name__

    # 以上代码不可使用，否则在单独调试指标时会出现循环引用的问题
    # 以下代码进行硬编码，策略函数名若被修改，需要手动修改以下代码，否则可能导致错误

    Boll_BLL = "_boll_strategy"
    SMA_MTL = "_mutiline_strategy"
    EMA_MTL = "_mutiline_strategy"
    SRLine_PSS = "_pressure_area_strategy"
    RSI_ECD = "_exceedingly_trade_strategy"
    RSI_VRS = "_versus_sma_strategy"
