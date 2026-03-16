from .vit import vit
from .dinov3 import dinov3

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vit':
        return vit(cfg)
    if cfg.MODEL.BACKBONE.TYPE == 'dinov3':
        return dinov3(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
