"""__init__.py"""
import sys
import os

# 解决模块导入问题
# 获取本文件当前路径
current_path = os.path.abspath(os.path.dirname(__file__))
# 获取项目父目录utils的路径
utils_path = os.path.abspath(os.path.join(current_path, ".."))
# 获取项目根目录goInvest的路径
goInvest_path = os.path.abspath(os.path.join(utils_path, ".."))
# 将项目根目录添加到Python路径中
sys.path.append(goInvest_path)
