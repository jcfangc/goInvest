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


class SeriesOperation(Enum):
    Smoother = "smoother"
    InflectionPoint = "inflectionPoint"
    Refine = "refine"
