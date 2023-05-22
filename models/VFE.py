from models.swin_transformer import SwinTransformer
from torch import nn


class VFE(nn.Module):
    def __init__(self, num_phase_class, num_tool_class):
        super().__init__()
        self.backbone = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                                        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                                        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.5,
                                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                        use_checkpoint=False)  # base
        self.fc_phase = nn.Linear(1024, num_phase_class)
        self.fc_tool = nn.Linear(1024, num_tool_class)

    def forward(self, input):
        xs = self.backbone(input)
        phase = self.fc_phase(xs)
        tool = self.fc_tool(xs)
        return phase, tool, xs

