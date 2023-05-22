from torch import nn
import torch
from models.miniGPT import GPT
from models.miniVAE import vae

class LAST(nn.Module):
    def __init__(self, num_phase_class=7, num_tool_class=7, band_width=[1000-1, 10-1]):
        super(LAST, self).__init__()
        self.proj = nn.Sequential(nn.Linear(1024, 512), nn.Dropout(p=0.5))
        self.TemporalNN = GPT(n_embd=512, n_layer=6, n_head=4, band_width=band_width)
        self.fc_phase = nn.Linear(512, num_phase_class)
        self.fc_tool = nn.Linear(512, num_tool_class)
        self.vae = vae(num_phase_class+num_tool_class, 3, 1)

    def forward(self, input, target=None, training=False):
        feat = self.proj(input)
        feat_phase, feat_tool = self.TemporalNN(feat)
        phase = self.fc_phase(feat_phase)
        tool = self.fc_tool(feat_tool)
        if training:
            phase_vae = torch.softmax(phase, dim=-1)
            tool_vae = torch.sigmoid(tool)
            pred_input = torch.cat((phase_vae, tool_vae), dim=-1)
            _, kl_loss = self.vae(pred_input, target)
        else:
            kl_loss = 0.0

        phase = phase.squeeze(0)
        tool = tool.squeeze(0)

        return phase, tool, kl_loss

