from uer.layers import str2embedding
from uer.encoders import str2encoder
from uer.targets import str2target
from loopedtransformer.looped_encoder import LoopedTransformerEncoder
from loopedtransformer.halting_looped_encoder import HaltingLoopedTransformerEncoder
from loopedtransformer.looped_model import LoopedModel


def register_looped_components():
    """向 UER 注册表外部注入循环编码器。

    依据《AI协作规则》，不修改 uer/encoders/__init__.py，
    改为在运行时向 str2encoder 字典追加新条目。
    """
    str2encoder["looped_transformer"] = LoopedTransformerEncoder
    str2encoder["halting_looped_transformer"] = HaltingLoopedTransformerEncoder


def build_looped_model(args):
    """与 uer.model_builder.build_model 等价，但使用 LoopedModel 外壳。

    组装流程不变，仍是 embedding、encoder、target 三部分，
    差异只在最外层用 LoopedModel 替换 Model。
    """
    register_looped_components()
    embedding = str2embedding[args.embedding](args, len(args.vocab))
    encoder = str2encoder[args.encoder](args)
    target = str2target[args.target](args, len(args.vocab))
    model = LoopedModel(args, embedding, encoder, target)
    return model
