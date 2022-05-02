from torch import nn
from src.models.audiomer.audiomer_modules import AudiomerEncoder


class AudiomerClassification(nn.Module):
    def __init__(self, *, config, kernel_sizes, num_classes, mlp_dim, num_heads, dim_head, depth, pool, mlp_dropout, use_residual, expansion_factor, input_size, use_attention, use_se, equal_strides):
        '''config[-1] will be the output sequence length'''
        assert(pool in ['none', "mean", "cls"])
        super().__init__()

        self.pool = pool
        self.use_cls = True if self.pool == "cls" else False

        self.encoder = AudiomerEncoder(
            config=config, kernel_sizes=kernel_sizes, num_heads=num_heads, depth=depth, use_residual=use_residual, use_cls=self.use_cls, dim_head=dim_head, expansion_factor=expansion_factor, input_size=input_size, use_attention=use_attention, use_se=use_se, equal_strides=equal_strides)

        self.classifier = nn.Sequential(
            nn.Linear(config[-1], mlp_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        if self.pool == "mean":
            x = x.mean(dim=2)
        else:
            x = x[:, :, 0]
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    labels = ['backward','bed','bird','cat','dog','down','eight','five','follow','forward','four','go','happy','house','learn','left','marvin','nine','no','off','on','one','right','seven','sheila','six','stop','three','tree','two','up','visual','wow','yes','zero']
    config = [1, 4, 8, 8, 16, 16, 32, 32, 64, 64]  # Audiomer S - 180K
    input_size = 8192 * 1
    model = AudiomerClassification(
        input_size=input_size,
        config=config,
        kernel_sizes=[5] * (len(config) - 1),
        num_classes=len(labels),
        depth=1,
        num_heads=2,
        pool="cls",
        mlp_dim=config[-1],
        mlp_dropout=0.2,
        use_residual=True,
        dim_head=32,
        expansion_factor=2,
        use_attention=True,
        use_se=True,
        equal_strides=False
    )#.cuda()

    inp = torch.randn(2, 1, input_size)#.cuda()
    print(model(inp).shape)
    summary(model, (1, input_size), device=device.type)
    count = 0
    for p in model.parameters():
        count += int(p.numel())
    print("# params: ", count)