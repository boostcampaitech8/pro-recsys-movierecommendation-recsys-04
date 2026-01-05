import torch
import torch.nn as nn


class BERT4Rec(nn.Module):
    """
    BERT4Rec with shared item embedding (weight tying)

    Token:
      PAD = 0
      item tokens = 1..num_items
      MASK = num_items + 1
    """

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_items = num_items
        self.max_len = max_len

        self.PAD = 0
        self.MASK = num_items + 1
        self.vocab_size = num_items + 2

        # embeddings
        self.item_embedding = nn.Embedding(
            self.vocab_size, hidden_size, padding_idx=self.PAD
        )
        self.position_embedding = nn.Embedding(max_len, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(hidden_size)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trm_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # output bias only (weight tying)
        self.output_bias = nn.Parameter(torch.zeros(self.vocab_size))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_bias)

    def forward(self, input_ids: torch.Tensor, pad_mask: torch.Tensor):
        """
        input_ids: (B, L)
        pad_mask : (B, L)  -> True at PAD positions
        """
        bsz, seq_len = input_ids.size()
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)

        x = self.item_embedding(input_ids) + self.position_embedding(pos)
        x = self.LayerNorm(self.dropout(x))

        h = self.trm_encoder(x, src_key_padding_mask=pad_mask)

        # shared embedding projection
        logits = torch.matmul(h, self.item_embedding.weight.t()) + self.output_bias
        return logits

    @torch.no_grad()
    def load_s3rec_item_embedding(self, s3rec_item_emb: torch.Tensor):
        """
        s3rec_item_emb: (num_items, hidden_size)
        BERT4Rec item token range is [1, num_items]
        """
        if s3rec_item_emb.shape[0] != self.num_items:
            raise ValueError(
                f"num_items mismatch: got {s3rec_item_emb.shape[0]} vs {self.num_items}"
            )
        if s3rec_item_emb.shape[1] != self.item_embedding.weight.shape[1]:
            raise ValueError(
                f"hidden_size mismatch: got {s3rec_item_emb.shape[1]} vs {self.item_embedding.weight.shape[1]}"
            )

        self.item_embedding.weight[
            1 : self.num_items + 1
        ].copy_(s3rec_item_emb.to(self.item_embedding.weight.device))
