#
# # -*- coding: utf-8 -*-
# # ---------------------
#
# import os
#
# PYTHONPATH = '..:.'
# if os.environ.get('PYTHONPATH', default=None) is None:
#     os.environ['PYTHONPATH'] = PYTHONPATH
# else:
#     os.environ['PYTHONPATH'] += (':' + PYTHONPATH)
#
# import yaml
# import socket
# import random
# import torch
# import numpy as np
# from path import Path
# from typing import Optional
# import termcolor
# from datetime import datetime
#
#
# def set_seed(seed=None):
#     # type: (Optional[int]) -> int
#     """
#     set the random seed using the required value (`seed`)
#     or a random value if `seed` is `None`
#     :return: the newly set seed
#     """
#     if seed is None:
#         seed = random.randint(1, 10000)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     return seed
#
#
# class Config(object):
#     HOSTNAME = socket.gethostname()
#     LOG_PATH = Path('./logs/')
#
#     def __init__(self, conf_file_path=None, seed=None, exp_name=None, log=True):
#         # type: (str, int, str, bool) -> None
#         """
#         :param conf_file_path: optional path of the configuration file
#         :param seed: desired seed for the RNG; if `None`, it will be chosen randomly
#         :param exp_name: name of the experiment
#         :param log: `True` if you want to log each step; `False` otherwise
#         """
#         #self.exp_name = exp_name
#         self.exp_name = exp_name if exp_name is not None else "default"
#         self.log_each_step = log
#
#         # print project name and host name
#         self.project_name = Path(__file__).parent.parent.basename()
#         m_str = f'┃ {self.project_name}@{Config.HOSTNAME} ┃'
#         u_str = '┏' + '━' * (len(m_str) - 2) + '┓'
#         b_str = '┗' + '━' * (len(m_str) - 2) + '┛'
#         print(u_str + '\n' + m_str + '\n' + b_str)
#
#         # define output paths
#         self.project_log_path = Path('./log')
#
#         # set random seed
#         self.seed = set_seed(seed)  # type: int
#
#         self.keys_to_hide = list(self.__dict__.keys()) + ['keys_to_hide']
#
#         # if the configuration file is not specified
#         # try to load a configuration file based on the experiment name
#         tmp = Path(__file__).parent / (self.exp_name + '.yaml')
#         if conf_file_path is None and tmp.exists():
#             conf_file_path = tmp
#
#         # read the YAML configuation file
#         if conf_file_path is None:
#             y = {}
#         else:
#             conf_file = open(conf_file_path, 'r')
#             y = yaml.load(conf_file, Loader=yaml.Loader)
#
#         # read configuration parameters from YAML file
#         # or set their default value
#         self.lr = y.get('lr', 0.0001)  # type: float
#         self.epochs = y.get('num_epochs', 100)  # type: int
#         self.n_workers = y.get('n_workers', 1)  # type: int
#         self.batch_size = y.get('batch_size', 64)  # type: int
#         self.quantiles = y.get('quantiles', [0.1, 0.5, 0.9])  # type: list
#         self.ds_name = y.get('ds_name', "electricity")  # type: str
#         self.all_params = y  # type: dict
#
#
#         self.exp_log_path = os.path.join("D:/PythonProject/chatbot/log", exp_name,
#                                          datetime.now().strftime("%m-%d-%Y - %H-%M-%S"))
#
#         default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.device = y.get('DEVICE', default_device)  # type: str
#
#     def write_to_file(self, out_file_path):
#         # type: (str) -> None
#         """
#         Writes configuration parameters to `out_file_path`
#         :param out_file_path: path of the output file
#         """
#         import re
#
#         ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
#         text = ansi_escape.sub('', str(self))
#         with open(out_file_path, 'w') as out_file:
#             print(text, file=out_file)
#
#     def __str__(self):
#         # type: () -> str
#         out_str = ''
#         for key in self.__dict__:
#             if key in self.keys_to_hide:
#                 continue
#             value = self.__dict__[key]
#             if type(value) is Path or type(value) is str:
#                 value = value.replace(Config.LOG_PATH, '$LOG_PATH')
#                 value = termcolor.colored(value, 'yellow')
#             else:
#                 value = termcolor.colored(f'{value}', 'magenta')
#             out_str += termcolor.colored(f'{key.upper()}', 'blue')
#             out_str += termcolor.colored(': ', 'red')
#             out_str += value
#             out_str += '\n'
#         return out_str[:-1]
#
#     def no_color_str(self):
#         # type: () -> str
#         out_str = ''
#         for key in self.__dict__:
#             value = self.__dict__[key]
#             if type(value) is Path or type(value) is str:
#                 value = value.replace(Config.LOG_PATH, '$LOG_PATH')
#             out_str += f'{key.upper()}: {value}\n'
#         return out_str[:-1]
#
#
# def show_default_params():
#     """
#     Print default configuration parameters
#     """
#     cnf = Config(exp_name='default')
#     print(f'\nDefault configuration parameters: \n{cnf}')
#
#
# if __name__ == '__main__':
#     show_default_params()

# -*- coding: utf-8 -*-
# ---------------------

import os
import yaml
import socket
import random
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import termcolor
import pandas as pd
from dataset.Dataset import Dataset
from dataset.Dataset_cesnet import CESNETDataset
from typing import Optional


def set_seed(seed=None):
    """
    Set the random seed using the required value (`seed`) or a random value if `seed` is `None`
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


class Config(object):
    HOSTNAME = socket.gethostname()
    LOG_PATH = Path("./logs/")

    def __init__(self, conf_file_path=None, seed=None, exp_name=None, log=True):
        """
        :param conf_file_path: YAML 配置文件路径
        :param seed: 训练随机种子
        :param exp_name: 实验名称
        :param log: 是否记录日志
        """
        self.exp_name = exp_name if exp_name is not None else "default"
        self.log_each_step = log

        # ✅ 打印项目名称
        self.project_name = Path(__file__).parent.parent.name
        print(f'┃ {self.project_name}@{Config.HOSTNAME} ┃')

        # ✅ 定义 log 路径
        self.project_log_path = Path("./log")
        self.exp_log_path = self.project_log_path / self.exp_name / datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        self.exp_log_path.mkdir(parents=True, exist_ok=True)  # 确保目录存在

        # ✅ 设置随机种子
        self.seed = set_seed(seed)

        # ✅ 确保 config.yaml 存在
        if conf_file_path is not None:
            conf_file_path = Path(conf_file_path)  # ✅ 转换为 Path 对象


        if conf_file_path is None or not conf_file_path.exists():
            print(f"⚠️ Warning: 配置文件 {conf_file_path} 不存在，将使用默认参数！")
            y = {}  # 使用默认配置
        else:
            with open(conf_file_path, 'r') as conf_file:
                y = yaml.safe_load(conf_file)
        # if conf_file_path is None:
        #     conf_file_path = Path(f"./config/{self.exp_name}.yaml")
        #
        # if not conf_file_path.exists():
        #     print(f"⚠️ {conf_file_path} 不存在，正在创建默认配置...")
        #     self.create_default_config(conf_file_path)
        #
        # # ✅ 读取 YAML 配置
        # with open(conf_file_path, "r") as conf_file:
        #     y = yaml.load(conf_file, Loader=yaml.Loader)

        # ✅ 读取或设置默认参数
        self.lr = y.get("lr", 0.0001)  # 学习率
        self.epochs = y.get("num_epochs", 100)  # 训练轮数
        self.n_workers = y.get("n_workers", 1)  # 数据加载线程数
        self.batch_size = y.get("batch_size", 64)  # 批量大小
        self.quantiles = y.get("quantiles", [0.1, 0.5, 0.9])  # 分位数
        self.ds_name = y.get("ds_name", "CSE-CIC-IDS2018")  # 数据集名称
        self.model = y.get("model", "tft")  # 选择的模型
        self.total_time_steps = y.get("total_time_steps", 96)  # 时间步长
        self.num_encoder_steps = y.get("num_encoder_steps", 48)  # 编码器步长
        self.output_size = y.get("output_size", 1)  # 预测输出大小
        self.all_params = y  # 保存所有参数

        self.num_heads = y.get("num_heads", 8)  # ✅ 默认为 8
        self.all_params["num_heads"] = self.num_heads  # ✅ 存入 all_params

        self.hidden_layer_size = y.get("hidden_layer_size", 160)  # 默认值160
        self.all_params["hidden_layer_size"] = self.hidden_layer_size

        self.dropout_rate = y.get("dropout_rate", 0.1)  # 默认值 0.1
        self.all_params["dropout_rate"] = self.dropout_rate  # 确保写入

        self.max_gradient_norm = y.get("max_gradient_norm", 5.0)  # 默认值 5.0
        self.all_params["max_gradient_norm"] = self.max_gradient_norm  # 确保写入 YAML

        self.early_stopping_patience = y.get("early_stopping_patience", 10)  # 默认值 10
        self.all_params["early_stopping_patience"] = self.early_stopping_patience  # 确保写入 YAML

        self.stack_size = y.get("stack_size", 1)  # 默认值 1
        self.all_params["stack_size"] = self.stack_size  # 确保写入 YAML

        dataset_path = r"D:\PythonProject\chatbot\dataset\preprocessed\CSE-CIC-IDS2018_1s\train.csv"
        dataset = Dataset(dataset_path)

        self.all_params["known_regular_inputs"]=dataset.known_regular_inputs
        self.all_params["static_input_loc"] =dataset.static_input_loc
        self.all_params["known_categorical_inputs"] = dataset.known_categorical_inputs
        self.all_params["input_obs_loc"]=dataset.input_obs_loc
        self.all_params["unknown_time_features"] = dataset.unknown_time_features


        dataset_path = r"D:\PythonProject\chatbot\dataset\preprocessed\CSE-CIC-IDS2018_1s\train.csv"
        try:
            self.df = pd.read_csv(dataset_path)


            category_counts = [
                #int(self.df["protocol"].nunique()),  # 计算 `protocol` 变量的类别数
                max(7, int(self.df["day_of_week"].nunique())),  # 计算 `is_holiday`
                max(2,int(self.df["is_weekend"].nunique())),
                max(2, int(self.df["is_business_hours"].nunique())),  # 计算 `day_of_week`
                #int(self.df["dst_port"].nunique())
            ]
            self.all_params["category_counts"] = category_counts  # 添加到 `all_params`
        except Exception as e:
            print(f"⚠️ 读取数据集时出错: {e}")
            self.all_params["category_counts"] = [-1]  # 默认值，防止失败

        # ✅ 设置默认设备
        self.device = y.get("DEVICE","cuda")


    def create_default_config(self, file_path):
        """
        创建默认的 config.yaml
        """
        default_config = {
            "lr": 0.0001,
            "num_epochs": 100,
            "n_workers": 1,
            "batch_size": 64,
            "quantiles": [0.1, 0.5, 0.9],
            "ds_name": "CSE-CIC-IDS2018",
            "total_time_steps": 96,
            "num_encoder_steps": 48,
            "output_size": 1,
            "DEVICE": "cuda",
        }

        file_path.parent.mkdir(parents=True, exist_ok=True)  # 确保 config 目录存在
        with open(file_path, "w") as f:
            yaml.dump(default_config, f)

        print(f"✅ 已创建默认 {file_path}")

    def __str__(self):
        """
        返回格式化后的配置信息
        """
        out_str = ""
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                value = value.as_posix()
            value = termcolor.colored(str(value), "yellow")
            out_str += termcolor.colored(f"{key.upper()}", "blue") + termcolor.colored(": ", "red") + value + "\n"
        return out_str.strip()

    import yaml
    import os

    # # ✅ 新增方法：将参数写入 YAML 文件
    # def write_to_file(self, out_file_path):
    #     """将配置参数写入 YAML 文件"""
    #     with open(out_file_path, 'w', encoding='utf-8') as file:
    #         yaml.dump(self.all_params, file, default_flow_style=False, allow_unicode=True)
    #     print(f"✅ 配置文件已成功写入: {out_file_path}")

    # def write_to_file(self, out_file_path):
    #     """将所有配置参数写入 YAML 文件"""
    #     with open(out_file_path, 'w', encoding='utf-8') as file:
    #         yaml.dump({k: v for k, v in self.__dict__.items() if not k.startswith("_")}, file, default_flow_style=False,
    #                   allow_unicode=True)
    #     print(f"✅ 配置文件已成功写入: {out_file_path}")

    # def write_to_file(self, out_file_path):
    #     """将所有配置参数写入 YAML 文件"""
    #     yaml_data = {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items() if not k.startswith("_")}
    #
    #     with open(out_file_path, 'w', encoding='utf-8') as file:
    #         yaml.dump(yaml_data, file, default_flow_style=False, allow_unicode=True)
    #
    #     print(f"✅ 配置文件已成功写入: {out_file_path}")
    def write_to_file(self, out_file_path):
        """🚀 只保存 `all_params` 的数据，不重复嵌套"""
        yaml_data = {
            "batch_size": self.batch_size,
            "category_counts": self.all_params["category_counts"],  # ✅ 直接保存 category_counts
            "device": self.device,
            "ds_name": self.ds_name,
            "epochs": self.epochs,
            "exp_log_path": str(self.exp_log_path),
            "exp_name": self.exp_name,
            "log_each_step": self.log_each_step,
            "lr": self.lr,
            "model": self.model,
            "n_workers": self.n_workers,
            "num_encoder_steps": self.num_encoder_steps,
            "output_size": self.output_size,
            "quantiles": self.quantiles,
            "seed": self.seed,
            "total_time_steps": self.total_time_steps,
            "num_heads": self.num_heads,  # ✅ 确保写入
            "static_input_loc": self.all_params["static_input_loc"],
            "known_regular_inputs": self.all_params["known_regular_inputs"],
            "known_categorical_inputs": self.all_params["known_categorical_inputs"],
            "input_obs_loc": self.all_params["input_obs_loc"],
            "unknown_time_features": self.all_params["unknown_time_features"]
        }

        # ✅ 只存储 `yaml_data`，确保 YAML 结构正确
        with open(out_file_path, 'w', encoding='utf-8') as file:
            yaml.dump(yaml_data, file, default_flow_style=False, allow_unicode=True)

        print(f"✅ 配置文件已成功写入: {out_file_path}")


if __name__ == '__main__':
    #cnf = Config(exp_name="CSE-CIC-IDS2018")
    cnf = Config(conf_file_path = r"D:\PythonProject\chatbot\config\config\CSE-CIC-IDS2018.yaml",
                 exp_name="CSE-CIC-IDS2018")

    # ✅ 指定 YAML 文件路径
    yaml_path = f"D:/PythonProject/chatbot/config/config/{cnf.exp_name}.yaml"

    # ✅ 调用 write_to_file 方法，将参数写入 YAML 文件
    cnf.write_to_file(yaml_path)

    print(f"✅ 配置文件已写入: {yaml_path}")


