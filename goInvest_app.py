from __init__ import goInvest_path
from productType import stock as sk

import pandas as pd


class goInvest:
    @staticmethod
    def main() -> None:
        # 分析的请求名单
        requirements = pd.read_excel(
            f"{goInvest_path}\\data\\requirement.xlsx",
            dtype={"identityCode": str},
            engine="openpyxl",
        )
        # 获取行数，shape函数返回值为元组(行数，列数)
        require_num = requirements.shape[0]

        # 遍历名单
        for sequence in range(0, require_num):
            match requirements.loc[sequence, "productType"]:
                case "stock":
                    stock = sk.Stock(requirements.loc[sequence])
                    stock.analyze_stock()


# 开始程序
goInvest.main()
