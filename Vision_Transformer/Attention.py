import torch
from torch import nn
from torch.functional import F
import numpy as np
import math

class Attention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, ..., q_len, q_dim): tensor containing projection vector for decoder.
        - **key** (batch, ..., k_len, k_dim): tensor containing features of the encoded input sequence. 
        - **value** (batch, ..., v_len, v_dim): tensor containing features of the encoded input sequence.
        - **mask** (batch, ..., q_len, k_len): tensor containing indices to be masked
        -  satisfy: q_dim = k_dim, v_len = k_len
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask=None):
        q_dim = query.size()[-1]
        k_dim = key.size()[-1]
        assert(q_dim == k_dim)
        score = torch.bmm(query, key.transpose(-2, -1)) # (batch, q_len, k_len):
        score = score/math.sqrt(k_dim)

        if mask is not None:
            score.masked_fill_(mask==0, -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn

class Multi_Head_Attention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_head):
        super(Multi_Head_Attention, self).__init__()
        assert (embed_dim%num_head == 0)
        self.num_head = num_head
        self.embed_dim = embed_dim
        self.head_dim = embed_dim//num_head

        self.Wv = nn.Linear(input_dim, embed_dim)
        self.Wk = nn.Linear(input_dim, embed_dim)
        self.Wq = nn.Linear(input_dim, embed_dim)

        self.atten = Attention()
        self.out_linear = nn.Linear(embed_dim, input_dim)


    def forward(self, q, k, v, mask):
        batch_size, seq_length, input_dim = v.size()

        v = self.Wv(v)
        k = self.Wk(k)
        q = self.Wv(q)

        # convert to (batch_size,seq_length,num_head, head_dim)
        v = v.reshape(batch_size, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3) # better to use reshape instead of view
        k = k.reshape(batch_size, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        q = q.reshape(batch_size, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        values, weights = self.atten(q, k, v, mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        values = self.out_linear(values)        

        return values, weights


def main():
    Q =  torch.tensor([[[ 0.3367,  0.1288],
        [ 0.2345,  0.2303],
        [-1.1229, -0.1863]]])

    K =  torch.tensor([[[ 2.2082, -0.6380],
        [ 0.4617,  0.2674],
        [ 0.5349,  0.8094]]])

    V =  torch.tensor([[[ 1.1103, -1.6898],
        [-0.9890,  0.9580],
        [ 1.3221,  0.8172]]])
    ATTN = Attention()
    values, attention = ATTN(Q, K, V)
    print("Values\n", values)
    print("Attention\n", attention)


if __name__ == "__main__":
    main()