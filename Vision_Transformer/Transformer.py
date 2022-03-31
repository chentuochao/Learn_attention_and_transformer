import torch
from torch import nn
from torch.functional import F
import numpy as np
import math
from Attention import Attention, Multi_Head_Attention
from torch.nn import CrossEntropyLoss, Dropout, Softmax, LayerNorm
from torch.nn.modules.utils import _pair
import torch.utils.data as data
import torch.optim as optim

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)
        
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)



class Encoder_Layer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_head, dim_feedforward, use_GELU = True, dropout = 0.0):
        super(Encoder_Layer, self).__init__()
        self.num_head = num_head
        self.embed_dim = embed_dim
        self.input_dim = input_dim

        self.multi_atten = Multi_Head_Attention(input_dim, embed_dim, num_head)

        if use_GELU:
            self.feedforward = nn.Sequential(
                nn.Linear(input_dim, dim_feedforward),
                nn.Dropout(dropout),
                GELU(),
                nn.Linear(dim_feedforward, input_dim)
            )
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(input_dim, dim_feedforward),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.Linear(dim_feedforward, input_dim)
            )     

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        ## x - (batch_size, len_seq, input_dim)
        atten_out, weights = self.multi_atten(x, x, x, mask)
        atten_out =  x + self.dropout(atten_out)
        x = self.norm1(atten_out)
        ## feedforward
        y = self.feedforward(x)
        y = x + self.dropout(y)
        y = self.norm2(y)

        return y



class Transformer_Encoder(nn.Module):
    def __init__(self, num_layer, input_dim, embed_dim, num_head, dim_feedforward, use_GELU = True, dropout = 0.0):
        super(Transformer_Encoder, self).__init__()

        self.transformer_blocks = nn.ModuleList(
            [Encoder_Layer(input_dim, embed_dim, num_head, dim_feedforward, use_GELU, dropout) for _ in range(num_layer)])
    
    def forward(self, x, mask=None):
        ## x - (batch_size, len_seq, input_dim)
        for l in self.transformer_blocks:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.multi_atten(x, mask=mask)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x


class VisionTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer, dim_feedforward
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        self.transformer = nn.Sequential(*[Encoder_Layer(embed_dim, embed_dim, num_heads, hidden_dim, True, dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))

    def calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.forward(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        #self.log(f'{mode}_loss', loss)
        #self.log(f'{mode}_acc', acc)
        if mode == "train":
            return loss
        else:
            return preds, loss, acc

    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out