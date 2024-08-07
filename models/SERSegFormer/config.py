from transformers.modeling_utils import PretrainedConfig
from typing import Optional

class SersegformerConfig(PretrainedConfig):
    model_type = "sersegformer"

    def __init__(
        self,
        image_size=512,
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256], # number of hidden states channels. [torch.Size([1, 32, 128, 128]), torch.Size([1, 64, 64, 64]), torch.Size([1, 160, 32, 32]), torch.Size([1, 256, 16, 16])] for (3,512,512) input
        downsampling_rates=[1, 4, 8, 16],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        drop_path_rate=0.1,
        layer_norm_eps=1e-6,
        decoder_hidden_size=256, # channels compression
        upsample=False,
        reshape_last_stage=True,
        abg=False, # Attention-boosting gate from https://arxiv.org/pdf/2401.15741
        dbn=False, # Dilation-based convolition network from https://arxiv.org/pdf/2401.15741
        afn=False, # Attention-fusion Networks
        critic=False, # Add critic loss
        **kwargs
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.depths = depths
        self.sr_ratios = sr_ratios
        self.hidden_sizes = hidden_sizes
        self.downsampling_rates = downsampling_rates
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.mlp_ratios = mlp_ratios
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.decoder_hidden_size = decoder_hidden_size
        self.reshape_last_stage = reshape_last_stage
        self.abg = abg
        self.dbn = dbn
        self.critic = critic
        self.upsample = upsample
        self.afn = afn
