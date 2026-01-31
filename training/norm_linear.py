import torch
import torch.nn as nn

class NormLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 ignore_nega_value,
                 use_layer_norm=True,
                 use_bias=True):
        super(NormLinear, self).__init__()
        self.norm = nn.LayerNorm(in_features) if use_layer_norm else nn.Identity()
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.ignore_nega_value = ignore_nega_value

    def forward(self, x):
        y = self.norm(x)
        y = self.linear(y)
        if self.ignore_nega_value:
            has_negative_per_step = (x < 0).any(dim=-1, keepdim=True)
            y = torch.where(has_negative_per_step, torch.zeros_like(y), y)
        return y

if __name__ == "__main__":
    test_input = [
        [
            [
                -1,0,3
            ]
        ],
        [
            [
                0,0,3
            ]
        ],
    ]
    test_input = torch.asarray(test_input,dtype=torch.float32)
    print(test_input.shape)
    m = NormLinear(in_features=3,out_features=3,ignore_nega_value=True)
    y = m.forward(test_input)
    print(y)
    print(y.shape)