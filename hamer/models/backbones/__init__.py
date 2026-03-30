from .vit import vit, vitpose_base
from .dinov3 import dinov3

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vit':
        return vit(cfg)
    if cfg.MODEL.BACKBONE.TYPE == 'dinov3':
        return dinov3(cfg)
    if cfg.MODEL.BACKBONE.TYPE == 'vitpose_base':
        return vitpose_base(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
