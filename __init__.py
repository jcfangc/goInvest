import sys
import os

# 解决模块导入问题
# 获取本文件当前路径
goInvest_path = os.path.abspath(os.path.dirname(__file__))
# 将项目根目录添加到Python路径中
sys.path.append(goInvest_path)


def folder_init_test() -> str:
    folder_path = os.path.join(goInvest_path, "data")
    if not os.path.exists(folder_path):
        return "data"

    folder_path = os.path.join(goInvest_path, "data", "kline")
    if not os.path.exists(folder_path):
        return "data\\kline"

    folder_path = os.path.join(goInvest_path, "data", "kline", "indicator")
    if not os.path.exists(folder_path):
        return "data\\kline\\indicator"

    folder_path = os.path.join(goInvest_path, "data", "kline", "indicator", "strategy")
    if not os.path.exists(folder_path):
        return "data\\kline\\indicator\\strategy"

    folder_path = os.path.join(goInvest_path, "utils")
    if not os.path.exists(folder_path):
        return "utils"

    folder_path = os.path.join(goInvest_path, "utils", "indicator")
    if not os.path.exists(folder_path):
        return "utils\\indicator"

    folder_path = os.path.join(goInvest_path, "productType")
    if not os.path.exists(folder_path):
        return "productType"

    return "OK"


folder_name = folder_init_test()
while folder_name != "OK":
    # 构建文件夹结构
    match folder_name:
        case "data":
            os.mkdir(os.path.join(goInvest_path, "data"))
            os.mkdir(os.path.join(goInvest_path, "data", "kline"))
            os.mkdir(os.path.join(goInvest_path, "data", "kline", "indicator"))
            os.mkdir(
                os.path.join(goInvest_path, "data", "kline", "indicator", "strategy")
            )

        case "data\\kline":
            os.mkdir(os.path.join(goInvest_path, "data", "kline"))
            os.mkdir(os.path.join(goInvest_path, "data", "kline", "indicator"))
            os.mkdir(
                os.path.join(goInvest_path, "data", "kline", "indicator", "strategy")
            )

        case "data\\kline\\indicator":
            os.mkdir(os.path.join(goInvest_path, "data", "kline", "indicator"))
            os.mkdir(
                os.path.join(goInvest_path, "data", "kline", "indicator", "strategy")
            )

        case "data\\kline\\indicator\\strategy":
            os.mkdir(
                os.path.join(goInvest_path, "data", "kline", "indicator", "strategy")
            )

        case "utils":
            os.mkdir(os.path.join(goInvest_path, "utils"))
            os.mkdir(os.path.join(goInvest_path, "utils", "indicator"))

        case "utils\\indicator":
            os.mkdir(os.path.join(goInvest_path, "utils", "indicator"))

        case "productType":
            os.mkdir(os.path.join(goInvest_path, "productType"))
