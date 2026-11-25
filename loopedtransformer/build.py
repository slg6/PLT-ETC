from uer.layers import str2embedding
from uer.encoders import str2encoder
from uer.targets import str2target
from loopedtransformer.looped_encoder import LoopedTransformerEncoder
from loopedtransformer.halting_looped_encoder import HaltingLoopedTransformerEncoder
from loopedtransformer.looped_model import LoopedModel


def register_looped_components():
    str2encoder["looped_transformer"] = LoopedTransformerEncoder
    str2encoder["halting_looped_transformer"] = HaltingLoopedTransformerEncoder


def build_looped_model(args):
    register_looped_components()
    embedding = str2embedding[args.embedding](args, len(args.vocab))
    encoder = str2encoder[args.encoder](args)
    target = str2target[args.target](args, len(args.vocab))
    model = LoopedModel(args, embedding, encoder, target)
    return model
