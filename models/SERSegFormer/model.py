""" PyTorch SegFormer+SERNet (SERSegFormer) model. """


import collections
import math

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F


from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer


import sys; sys.path.append('../../')
from models.SERSegFormer.config import SersegformerConfig
from modules.abg import AttentionBoostingGate
from modules.dbn import DilationBasedNetwork
from modules.afn import ContextAFN, SpatialAFN
from modules.upsample import UpsampleBlock
from modules.dam import DAM


# torch.autograd.set_detect_anomaly(True)

# AbG layers
# SersegformerLayer -> SersegformerMixFFN
# SersegformerOverlapPatchEmbeddings
# SersegformerEfficientSelfAttention




# Inspired by
# https://github.com/rwightman/pytorch-image-models/blob/b9bd960a032c75ca6b808ddeed76bee5f3ed4972/timm/models/layers/helpers.py
# From PyTorch internals
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


# Stochastic depth implementation
# Taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). This is the same as the
    DropConnect impl I created for EfficientNet, etc networks, however, the original name is misleading as 'Drop
    Connect' is a different form of dropout in a separate paper... See discussion:
    https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the layer and
    argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SersegformerOverlapPatchEmbeddings(nn.Module):
    """Construct the patch embeddings from an image."""

    def __init__(self, image_size, patch_size, stride, num_channels, hidden_size, abg):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        self.height, self.width = image_size[0] // patch_size[0], image_size[1] // patch_size[1]
        self.num_patches = self.height * self.width
        self.proj = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.abg = abg

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        x = self.proj(pixel_values)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.layer_norm(x)
        if self.abg:
            x = x + x*F.sigmoid(x)
        return x, height, width


class SersegformerEfficientSelfAttention(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads, sr_ratio, abg):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.abg = abg
        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr = nn.Conv2d(hidden_size, hidden_size, kernel_size=sr_ratio, stride=sr_ratio)
            self.layer_norm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        height,
        width,
        output_attentions=False,
    ):
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = hidden_states.shape
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            hidden_states = self.sr(hidden_states)
            hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            hidden_states = self.layer_norm(hidden_states)
            if self.abg:
                hidden_states = hidden_states + hidden_states*F.sigmoid(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class SersegformerSelfOutput(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SersegformerAttention(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads, sr_ratio):
        super().__init__()
        self.self = SersegformerEfficientSelfAttention(
            config=config, hidden_size=hidden_size, num_attention_heads=num_attention_heads, sr_ratio=sr_ratio, abg=config.abg
        )
        self.output = SersegformerSelfOutput(config, hidden_size=hidden_size)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, height, width, output_attentions=False):
        self_outputs = self.self(hidden_states, height, width, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class SersegformerDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, hidden_states, height, width):
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, num_channels, height, width)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        return hidden_states


class SersegformerMixFFN(nn.Module):
    def __init__(self, config, in_features, hidden_features=None, out_features=None, abg=False):
        super().__init__()
        out_features = out_features or in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.dwconv = SersegformerDWConv(hidden_features)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.abg = abg
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, height, width):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dwconv(hidden_states, height, width)
        if self.abg:
            hidden_states_add = F.sigmoid(hidden_states)*hidden_states
        hidden_states = self.intermediate_act_fn(hidden_states)
        if self.abg:
            hidden_states = hidden_states + hidden_states_add
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SersegformerLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config, hidden_size, num_attention_heads, drop_path, sr_ratio, mlp_ratio, abg):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.attention = SersegformerAttention(
            config, hidden_size=hidden_size, num_attention_heads=num_attention_heads, sr_ratio=sr_ratio
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = SersegformerMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size, abg=abg)

    def forward(self, hidden_states, height, width, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states),  # in Segformer, layernorm is applied before self-attention
            height,
            width,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection (with stochastic depth)
        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states

        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        # second residual connection (with stochastic depth)
        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hidden_states

        outputs = (layer_output,) + outputs

        return outputs

from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class SerSegEncoderOutput(BaseModelOutput):
    depths: Optional[Tuple[torch.FloatTensor]] = None

class SersegformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # patch embeddings
        embeddings = []
        for i in range(config.num_encoder_blocks):
            n_channels = config.num_channels if i == 0 else config.hidden_sizes[i - 1]
            if self.config.add_depth_channel:
                n_channels += 1

            embeddings.append(
                SersegformerOverlapPatchEmbeddings(
                    image_size=config.image_size // config.downsampling_rates[i],
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=n_channels, # in channels
                    hidden_size=config.hidden_sizes[i], # out channels
                    abg=self.config.abg
                )
            )
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    SersegformerLayer(
                        config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=dpr[cur + j],
                        sr_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                        abg=self.config.abg
                    )
                )
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

        # Layer norms
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)]
        )

    def forward(
        self,
        pixel_values,
        depth=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]

        if self.config.add_depth_channel:
            assert depth is not None
            depths = [depth]
            for sr in self.config.downsampling_rates[1:]:
                depths.append(F.interpolate(depth, scale_factor=1/sr, mode='bilinear', align_corners=False))
        else:
            depths = [None for _ in self.config.downsampling_rates]

        hidden_states = pixel_values
        for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm, depths)):
            embedding_layer, block_layer, norm_layer, depth = x
            # first, obtain patch embeddings
            if self.config.add_depth_channel:
                hidden_states = torch.cat([hidden_states, depth], dim=1)
            hidden_states, height, width = embedding_layer(hidden_states)
            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            # third, apply layer norm
            hidden_states = norm_layer(hidden_states)
            # fourth, optionally reshape back to (batch_size, num_channels, height, width)
            if idx != len(self.patch_embeddings) - 1 or (
                idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
            ):
                hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)



        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return SerSegEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            depths=tuple(depths)
        )


class SersegformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SersegformerConfig
    base_model_prefix = "sersegformer"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



class SersegformerModel(SersegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = SersegformerEncoder(config)

        self.init_weights()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(self, pixel_values, depth=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values,
            depth=depth,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return SerSegEncoderOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            depths=encoder_outputs.depths
        )




class SersegformerForImageClassification(SersegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.segformer = SersegformerModel(config)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_sizes[-1], config.num_labels)

        self.init_weights()


    def forward(
        self,
        pixel_values=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # reshape last hidden states to (batch_size, height*width, hidden_size)
        batch_size = sequence_output.shape[0]
        sequence_output = sequence_output.reshape(batch_size, -1, self.config.hidden_sizes[-1])

        # global average pooling
        sequence_output = sequence_output.mean(dim=1)

        logits = self.classifier(sequence_output)
        return logits

        # loss = None
        # if labels is not None:
        #     if self.num_labels == 1:
        #         #  We are doing regression
        #         loss_fct = MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )



class SersegformerMLP(nn.Module):
    """
    Linear Embedding.
    """
    def __init__(self, config: SersegformerConfig, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states



class SersegformerDecodeHead(SersegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        self.config = config
        mlps = []
        hidden_dims = [x for x in config.hidden_sizes]
        if config.dbn:
            hidden_dims[0] *= 2
            hidden_dims[1] *= 2
        if config.afn:
            hidden_dims[0] *= 2
            hidden_dims[1] *= 2
            hidden_dims[2] = 2*hidden_dims[2]+hidden_dims[0]
            hidden_dims[3] = 2*hidden_dims[3]+hidden_dims[1]

        if self.config.add_depth_channel:
            for i in range(len(hidden_dims[:-1])):
                hidden_dims[i] += 1

        for hidden_dim in hidden_dims:
            mlp = SersegformerMLP(config, input_dim=hidden_dim)
            mlps.append(mlp)
        
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        # self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)
        

        if self.config.upsample:
            self.upsample1 = UpsampleBlock(config.decoder_hidden_size, config.decoder_hidden_size//4)
            self.upsample2 = UpsampleBlock(config.decoder_hidden_size//4, config.decoder_hidden_size//4)
            self.classifier = nn.Linear(config.decoder_hidden_size//4, config.num_labels)
        else:
            self.classifier = nn.Linear(config.decoder_hidden_size, config.num_labels)

        if self.config.dam:
            self.dam = DAM(hidden_dims[0])
        
        if self.config.dbn: # Applied to the first hidden state (which is spatial)
            self.dbn1 = DilationBasedNetwork(config.hidden_sizes[0])
            self.dbn2 = DilationBasedNetwork(config.hidden_sizes[1])

        if self.config.afn:
            self.afn11 = SpatialAFN(config.hidden_sizes[0] if not self.config.dbn else 2*config.hidden_sizes[0])
            self.afn13 = ContextAFN(2*config.hidden_sizes[0] if not self.config.dbn else 4*config.hidden_sizes[0])
            self.afn22 = SpatialAFN(config.hidden_sizes[1] if not self.config.dbn else 2*config.hidden_sizes[1])
            self.afn24 = ContextAFN(2*config.hidden_sizes[1] if not self.config.dbn else 4*config.hidden_sizes[1])
            self.afn33 = SpatialAFN(config.hidden_sizes[2])
            self.afn44 = SpatialAFN(config.hidden_sizes[3])

            if self.config.upsample:
                self.afn_final = nn.Sequential(ContextAFN(config.decoder_hidden_size//4),
                                               UpsampleBlock(config.decoder_hidden_size//4, config.decoder_hidden_size//4, stride=4))


    def forward(self, encoder_hidden_states, depths=None):
        batch_size, _, _, _ = encoder_hidden_states[-1].shape
        all_hidden_states = ()
        encoder_hidden_states = list(encoder_hidden_states)
        if self.config.dbn: # apply to encoder_hidden_states[0]
            encoder_hidden_states[0] = torch.cat([self.dbn1(encoder_hidden_states[0]), encoder_hidden_states[0]], dim=1)
            encoder_hidden_states[1] = torch.cat([self.dbn2(encoder_hidden_states[1]), encoder_hidden_states[1]], dim=1)

            # [config.hidden_sizes[0], config.hidden_sizes[0]+config.hidden_sizes[1], config.hidden_sizes[0]+config.hidden_sizes[1]+config.hidden_sizes[2], config.hidden_sizes[3]]
        
        if self.config.afn:
            # Apply attention-fusion networks. SpatialAFN to encoder_hidden_states[0] and encoder_hidden_states[1], ContentAFN to encoder_hidden_states[2] and encoder_hidden_states[3]
            encoder_hidden_states[0] = torch.cat([self.afn11(encoder_hidden_states[0]), encoder_hidden_states[0]], dim=1)
            encoder_hidden_states[1] = torch.cat([self.afn22(encoder_hidden_states[1]), encoder_hidden_states[1]], dim=1)
            encoder_hidden_states[2] = torch.cat([self.afn13(encoder_hidden_states[0]), self.afn33(encoder_hidden_states[2]), encoder_hidden_states[2]], dim=1)
            encoder_hidden_states[3] = torch.cat([self.afn24(encoder_hidden_states[1]), self.afn44(encoder_hidden_states[3]), encoder_hidden_states[3]], dim=1)
            
        if self.config.add_depth_channel:
            for i, depth in enumerate(depths[1:]):
                encoder_hidden_states[i] = torch.cat([encoder_hidden_states[i], depth], dim=1)

        if self.config.dam:
            encoder_hidden_states[0] = self.dam(encoder_hidden_states[0])

        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate( # TODO make smarter upsampling
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)


        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        
        if self.config.upsample: # Upsample by 4
            hidden_states = self.upsample1(hidden_states)
            if self.config.afn:
                hidden_states = hidden_states + self.afn_final(hidden_states)
            hidden_states = self.upsample2(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        # logits = self.classifier(hidden_states)
        logits = self.classifier(hidden_states.permute(0,2,3,1)).permute(0,3,1,2)

        return logits




class SersegformerForSemanticSegmentation(SersegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.sersegformer = SersegformerModel(config)
        self.decode_head = SersegformerDecodeHead(config)

        self.init_weights()


    def forward(
        self,
        pixel_values,
        depth=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.sersegformer(
            pixel_values,
            depth=depth,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )
        
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        depths = outputs.depths

        logits = self.decode_head(encoder_hidden_states, depths)
        return logits

        # loss = None
        # if labels is not None:
        #     if self.config.num_labels == 1:
        #         raise ValueError("The number of labels should be greater than one")
        #     else:
        #         # upsample logits to the images' original size
        #         upsampled_logits = nn.functional.interpolate(
        #             logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        #         )
        #         loss_fct = CrossEntropyLoss(ignore_index=255)
        #         loss = loss_fct(upsampled_logits, labels)

        # if not return_dict:
        #     if output_hidden_states:
        #         output = (logits,) + outputs[1:]
        #     else:
        #         output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states if output_hidden_states else None,
        #     attentions=outputs.attentions,
        # )


if __name__ == '__main__':
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    config = SersegformerConfig(abg=True,
                                num_labels=1,
                                # decoder_hidden_size=128, # Channels compression (-4ms)
                                upsample=True,
                                afn=False,
                                add_depth_channel=True,
                                dbn=True, # Cheap, no inference time increase,
                                dam=True
                                )
    model = SersegformerForSemanticSegmentation(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {n_params}')
    dummy_input = torch.randn((1,3,512,512)).to(device)
    dummy_depth = torch.randn((1,1,512,512)).to(device)
    print(model(dummy_input, dummy_depth).shape)

    model.eval()
    for _ in range(10):
        model(dummy_input, dummy_depth)
    n_eval_iters = 100
    
    start_ts = time.time()
    with torch.no_grad():
        for _ in range(n_eval_iters):
            output = model(dummy_input, dummy_depth)
    print(output.shape)
    print(f'Inference_time (ms): {1000*(time.time()-start_ts)/n_eval_iters}\nDevice: {str(device)}')
