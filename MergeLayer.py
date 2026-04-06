class PatchMerging(nn.Module):
    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x


class PatchMerging_new(nn.Module):
    def __init__(self, dim: int, attn_channels: int = None, attn_channels_ratio: float = 0.5, kernel_size: int = 3):
        super().__init__()
        self.attn_channels = attn_channels or int(dim * attn_channels_ratio)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, 2, kernel_size // 2, bias=False, groups=dim)
        self.src_norm = nn.RMSNorm(self.attn_channels)
        self.tgt_norm = nn.RMSNorm(self.attn_channels)
        self.qg_proj = nn.Linear(self.attn_channels, self.attn_channels * 2, bias=True)
        self.kv_proj = nn.Linear(self.attn_channels, self.attn_channels * 2, bias=True)
        self.out_proj = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim * 2)
        self.pos_embed = nn.Parameter(torch.zeros(4, self.attn_channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.register_buffer('scale', torch.tensor(1.0 / (self.attn_channels ** 0.5)))

    
    def forward(self, x):
        B, C, H, W = x.shape
        src = x[:, :self.attn_channels].reshape(B, self.attn_channels, H//2, 2, W//2, 2).permute(0, 2, 4, 3, 5, 1).flatten(-3, -2) + self.pos_embed  # shape: [B, H//2, W//2, 4, C_attn]
        y = self.dwconv(x)  # shape: [B, C, H//2, W//2]
        tgt = y[:, :self.attn_channels].permute(0, 2, 3, 1).unsqueeze(-2)  # shape: [B, H//2, W//2, 1, C_attn]
        k, v = self.kv_proj(self.src_norm(src)).chunk(2, dim=-1)  # shape: [B, H//2, W//2, 4, C_attn]
        q, g = self.qg_proj(self.tgt_norm(tgt)).chunk(2, dim=-1)  # shape: [B, H//2, W//2, 1, C_attn]
        attn_out = ((q @ k.transpose(-1, -2) * self.scale).softmax(dim=-1) @ v) * torch.sigmoid(g)  # shape: [B, H//2, W//2, 1, C_attn]
        attn_out = (attn_out + tgt).squeeze(-2).permute(0, 3, 1, 2)  # shape: [B, C_attn, H//2, W//2]
        out = self.bn(self.out_proj(torch.cat([attn_out, y[:, self.attn_channels:]], dim=1)))  # shape: [B, 2 * C, H//2, W//2]

        return out
