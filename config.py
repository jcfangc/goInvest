"""
文件描述：
1. 提供项目根目录路径__BASE_PATH__
2. 日志记录
3. 提供管理目录的类，并进行目录管理
"""

import os
import sys
import logging
import pandas as pd
import json
import datetime as dt
from pandas import DataFrame
from typing import Optional, List

# 获取本文件当前路径
__BASE_PATH__ = os.path.abspath(os.path.dirname(__file__))
# 将项目根目录添加到Python路径中
sys.path.append(__BASE_PATH__)


def do_logging() -> logging.Logger:
    """日志记录"""

    # 创建日志记录器
    logger = logging.getLogger(__name__)
    # 设置日志记录级别
    logger.setLevel(logging.DEBUG)
    # 创建日志格式化器
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(lineno)d\n%(message)s"
    )
    # 创建流处理器并添加到日志记录器中
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # 创建文件处理器并添加到日志记录器中
    md = dt.date.today().strftime("%m%d")
    file_handler = logging.FileHandler(f"{__BASE_PATH__}\\app_{md}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # 删除过往的日志文件
    for file_name in os.listdir(__BASE_PATH__):
        if file_name.endswith(".log") and (file_name.endswith(f"{md}.log") is False):
            os.remove(os.path.join(__BASE_PATH__, file_name))

    return logger


class DirectoryNode:
    def __init__(self, name: str, parent: Optional["DirectoryNode"] = None) -> None:
        self.name = name
        self.parent = parent
        self.children: List[DirectoryNode] = []

        # 若父节点不为空，则将自身添加到父节点的子节点列表中，保证父节点和子节点的同步
        if parent is not None and (self.is_parent_child() is False):
            # 将自身添加到父节点的子节点列表中
            parent.add_child(self)

    def is_parent_child(self) -> bool:
        # 检查父节点的子节点列表中是否已经存在本节点
        if self.parent is not None:
            for exist_child in self.parent.children:
                if exist_child.name == self.name:
                    return True
            return False

        # 若父节点为空，说明本身是根节点
        else:
            return True

    def has_child(self, child_name: str) -> Optional["DirectoryNode"]:
        # 检测是否存在同名子节点
        for exist_child in self.children:
            if exist_child.name == child_name:
                return exist_child
        return None

    def add_child(self, child: "DirectoryNode") -> None:
        if self.has_child(child.name) is None:
            child.parent = self
            self.children.append(child)

    def create_subdirectory(self, new_dir_name: str) -> "DirectoryNode":
        # 检测是否存在同名子目录
        exist_dir = self.has_child(new_dir_name)
        if exist_dir is not None:
            return exist_dir
        # 以防new_dir_name不在子节点列表中，但是在文件夹中
        elif exist_dir is None and os.path.isdir(
            os.path.join(self.full_name(), new_dir_name)
        ):
            new_child = DirectoryNode(new_dir_name, parent=self)
            self.add_child(new_child)
            return new_child

        # 创建子目录
        os.makedirs(os.path.join(self.full_name(), new_dir_name))
        new_child = DirectoryNode(new_dir_name, parent=self)
        self.add_child(new_child)
        return new_child

    def full_name(self) -> str:
        # 递归调用，直到根目录
        if self.parent is None:
            return self.name
        else:
            return os.path.join(self.parent.full_name(), self.name)

    def to_dict(self) -> dict:
        if self.children != []:
            node_dict = {self.name: {}}
            for child in self.children:
                node_dict[self.name].update(child.to_dict())
            return node_dict
        else:
            return {self.name: {}}


class DirectoryManager:
    def __init__(self):
        self.goinvest = DirectoryNode(name=f"{__BASE_PATH__}")
        self.data = self.goinvest.create_subdirectory("data")
        self.data_full = self.data.full_name()
        self.requirement = self.read_requirement_from_data_dir()
        self.check_config_json()
        self.logger = do_logging()

    def check_config_json(self) -> None:
        """检查config.json文件是否存在"""

        # 检查config.json文件是否存在
        if os.path.isfile(f"{__BASE_PATH__}\\config.json"):
            # 读取现有 JSON 文件
            with open(f"{__BASE_PATH__}\\config.json", "r") as f:
                self.config_data = json.load(f)
        else:
            # 创建config.json文件
            with open(f"{__BASE_PATH__}\\config.json", "w") as f:
                f.write("{}")

    def read_requirement_from_data_dir(self) -> DataFrame:
        """从data目录下的requirement.xlsx中读取需求"""

        # 尝试访问data下的requirement.xlsx
        if os.path.isfile(f"{self.data_full}\\requirement.xlsx"):
            # 读取requirement.xlsx
            requirement = pd.read_excel(
                f"{self.data_full}\\requirement.xlsx",
                dtype={"identityCode": str},
                engine="openpyxl",
            )
            return requirement
        else:
            # 创建excel文件
            requirement = pd.DataFrame(columns=["productType", "identityCode"])
            requirement.to_excel(
                f"{self.data_full}\\requirement.xlsx",
                index=False,
                engine="openpyxl",
            )
            return requirement

    def construct_dir_from_requirement(self) -> None:
        """根据requirement.xlsx创建子目录"""

        product_type_list = self.requirement["productType"].unique()
        for product_type in product_type_list:
            # 根据产品类型创建子目录
            self.data.create_subdirectory(product_type)
            # 在对应产品类型的子目录下创建子目录
            identity_code_list = self.requirement[
                self.requirement["productType"] == product_type
            ]["identityCode"]
            for identity_code in identity_code_list:
                identity_dir = self.data.children[-1].create_subdirectory(identity_code)
                # 在对应产品下创建固定的kline、indicator、strategy子目录
                identity_dir.create_subdirectory("kline")
                identity_dir.create_subdirectory("indicator")
                identity_dir.create_subdirectory("strategy")
                identity_dir.create_subdirectory("test")

    def update_data_directory_structure_to_json(self) -> None:
        """将data目录下的目录结构更新到config.json文件中"""

        if "dataDirectoryStructure" not in self.config_data.keys():
            # 创建 "dataDirectoryStructure" 键
            self.config_data["dataDirectoryStructure"] = {}

        # 更新 "dataDirectoryStructure" 键的内容
        self.config_data["dataDirectoryStructure"] = self.goinvest.to_dict()
        # 保存回文件
        with open(f"{__BASE_PATH__}\\config.json", "w") as f:
            json.dump(self.config_data, f, indent=4)

    def struct_compare_with_requirement(self) -> bool:
        """检查data目录下的目录结构是否与requirement中的目录结构一致"""

        def check_directory_exist(path):
            """辅助函数：检查目录是否存在"""
            if not os.path.isdir(path):
                self.logger.debug(f"目录 {path} 不存在")
                return False
            return True

        # 检查产品类型目录是否存在
        for product_type in self.requirement["productType"].unique():
            product_path = os.path.join(self.data_full, product_type)
            if not check_directory_exist(product_path):
                return False

            # 检查具体产品目录是否存在
            for identity_code in self.requirement[
                self.requirement["productType"] == product_type
            ]["identityCode"]:
                identity_path = os.path.join(product_path, identity_code)
                if not check_directory_exist(identity_path):
                    return False

                # 检查具体产品目录下的子目录是否存在
                for sub_dir in ["kline", "indicator", "strategy", "test"]:
                    sub_dir_path = os.path.join(identity_path, sub_dir)
                    if not check_directory_exist(sub_dir_path):
                        return False

        return True

    def directoty_manage(self) -> DataFrame:
        """主函数：目录管理"""

        # 检查data目录下的目录结构是否与requirement中的目录结构一致
        if self.struct_compare_with_requirement() is True:
            return self.requirement

        # 根据requirement.xlsx创建子目录
        self.construct_dir_from_requirement()
        # 生成json文件
        self.update_data_directory_structure_to_json()

        # 返回requirement
        return self.requirement
