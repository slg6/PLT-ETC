"""循环 Transformer 流量预训练模块。

本目录是与 traffictransformer 平级的新代码，遵循《AI协作规则》，
不修改 traffictransformer 下的任何原始文件，通过继承与外部注册复用原有组件。
"""

from loopedtransformer.looped_encoder import LoopedTransformerEncoder
from loopedtransformer.looped_model import LoopedModel
from loopedtransformer.build import register_looped_components, build_looped_model

__all__ = [
    "LoopedTransformerEncoder",
    "LoopedModel",
    "register_looped_components",
    "build_looped_model",
]
