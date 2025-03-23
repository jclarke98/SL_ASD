import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import math

class Encoder(torch.nn.Module):
    def __init__(self, voice_size=192, face_size=512, embedding_size=128, shared=True):
        super(Encoder, self).__init__()
        # input->drop-fc256-relu-[fc256-relu-fc128]
        mid_dim = 256

        def create_front(input_size):
            return torch.nn.Sequential(
                torch.nn.Dropout(),
                torch.nn.Linear(input_size, mid_dim),
                torch.nn.ReLU(),
            )

        def create_rare():
            return torch.nn.Sequential(
                torch.nn.Linear(mid_dim, mid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(mid_dim, embedding_size),
            )

        face_rare = create_rare()
        if shared:
            voice_rare = face_rare
        else:
            voice_rare = create_rare()

        self.face_encoder = torch.nn.Sequential(
            create_front(face_size),
            face_rare
        )
        self.voice_encoder = torch.nn.Sequential(
            create_front(voice_size),
            voice_rare
        )

    def forward(self, voice_data, face_data):
        v_emb = self.voice_encoder(voice_data)
        f_emb = self.face_encoder(face_data)
        return v_emb, f_emb


class TransformerEncoder(nn.Module):
    """
    A Transformer Encoder.

    Args:
        d_model (int): The dimensionality of the input and output tensors.
        nhead (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        nlayers (int, optional): The number of layers in the encoder. Defaults to 6.
    """

    def __init__(self, d_model, nheads=4, dropout=0.1, nlayers=4):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([self._make_layer(d_model, nheads, dropout) for _ in range(nlayers)])

    def _make_layer(self, d_model, nhead, dropout):
        return AttentionLayer(d_model, nhead, dropout)

    def forward(self, src, src_key_padding_mask=None):
        """
        Performs the forward pass of the Transformer Encoder.

        Args:
            src (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        for layer in self.layers:
            src, _ = layer(src, src, key_padding_mask=src_key_padding_mask)
        return src

class AttentionLayer(nn.Module):
    """
    A multi-head attention layer.

    Args:
        d_model (int): The dimensionality of the input and output tensors.
        nhead (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
    """

    def __init__(self, d_model, nhead, dropout=0.1, is_cross=False):
        super(AttentionLayer, self).__init__()
        self.attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

        self.is_cross = is_cross

    def forward(self, tar, src, key_padding_mask=None):
        """
        Performs the forward pass of the attention layer.
        """    
        tar = tar.transpose(0, 1)  # B, T_q, C -> T_q, B, C  # shape: (T_q, B, C)
        src = src.transpose(0, 1)  # B, T_kv, C -> T_kv, B, C  # shape: (T_kv, B, C)
        src2, attn_weights = self.attn(tar, src, src, key_padding_mask=key_padding_mask)  # shape: (T_q, B, C)
        tar = tar + self.dropout1(src2)  # shape: (T_q, B, C)
        tar = self.norm1(tar)  # shape: (T_q, B, C)
        tar2 = self.linear2(self.dropout(self.activation(self.linear1(tar))))  # shape: (T_q, B, C)
        tar = tar + self.dropout2(tar2)  # shape: (T_q, B, C)
        tar = self.norm2(tar)  # shape: (T_q, B, C)
        tar = tar.transpose(0, 1)  # T_q, B, C -> B, T_q, C  # shape: (B, T_q, C)
        return tar, attn_weights

class Decoder(torch.nn.Module):
    def __init__(self, embed_dim = 128, num_heads=4, num_layers=1, dropout=0.2,
                 num_heads_cross=4, visible_only=False):
        super(Decoder, self).__init__()
        
        # create instance of TansformerEncoder
        self.transformer = TransformerEncoder(embed_dim, 
                                              nheads=num_heads, 
                                              nlayers=num_layers, 
                                              dropout=dropout)

        # create instance of cross attention
        self.cross_attention = AttentionLayer(embed_dim, num_heads_cross, 
                                              dropout=dropout,
                                              is_cross=True)

        # create instance of linear layer
        self.fc = nn.Linear(embed_dim, 1)
        
    def forward_transformer(self, x, mask=None):
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.mean(dim=1).unsqueeze(1)
        return x
    
    def forward_cross_att(self, q, k_v, key_padding_mask=None):
        return self.cross_attention(q, k_v, key_padding_mask=key_padding_mask)

    def forward_linear(self, x):
        return self.fc(x)
    
    def forward(self, x, k_v, mask=None): # original: transformer -> average -> cross-attention -> linear
        x = self.forward_transformer(x, mask=mask).transpose(0,1)
        x, attn_scores = self.forward_cross_att(x, k_v)
        x = self.forward_linear(x)
        return x.squeeze(-1)