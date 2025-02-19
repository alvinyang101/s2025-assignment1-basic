import torch
import torch.nn as nn

from ece496b_basics.adapters import run_transformer_lm

class CustomModule(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, device, attn_pdrop=0.1, residual_pdrop=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.token_embeddings = nn.Embedding(vocab_size, d_model, dtype=torch.float32, device=device)
        self.position_embeddings = nn.Embedding(context_length, d_model, dtype=torch.float32, device=device)
        self.layers = nn.ModuleDict({
            f"{layer}": nn.ModuleDict({
                "attn": nn.ModuleDict({
                    "q_proj": nn.Linear(d_model, d_model, bias=False, device=device),
                    "k_proj": nn.Linear(d_model, d_model, bias=False, device=device),
                    "v_proj": nn.Linear(d_model, d_model, bias=False, device=device),
                    "output_proj": nn.Linear(d_model, d_model, bias=False, device=device),
                }),
                "ffn": nn.ModuleDict({
                    "w1": nn.Linear(d_model, d_ff, bias=False, device=device),
                    "w2": nn.Linear(d_ff, d_model, bias=False, device=device),
                }),
                "ln1": nn.LayerNorm(d_model, device=device),
                "ln2": nn.LayerNorm(d_model, device=device),
            }) for layer in range(num_layers)
        })

        self.ln_final = nn.LayerNorm(d_model, device=device)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device)

    def forward(self, inputs):
        outputs = run_transformer_lm(  # Call transformer_lm directly
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            attn_pdrop=self.attn_pdrop,  # Change if needed
            residual_pdrop=self.residual_pdrop,  # Change if needed
            weights=self.get_weight_dict(),  # New helper method
            in_indices=inputs
        )
        return outputs

    def get_weight_dict(self):
        """Creates a dictionary mapping parameter names to actual trainable tensors."""
        weight_dict = {}
        
        # Token and position embeddings
        weight_dict["token_embeddings.weight"] = self.token_embeddings.weight
        weight_dict["position_embeddings.weight"] = self.position_embeddings.weight

        # Transformer layers
        for layer in self.layers.keys():
            weight_dict[f"layers.{layer}.attn.q_proj.weight"] = self.layers[layer]["attn"]["q_proj"].weight
            weight_dict[f"layers.{layer}.attn.k_proj.weight"] = self.layers[layer]["attn"]["k_proj"].weight
            weight_dict[f"layers.{layer}.attn.v_proj.weight"] = self.layers[layer]["attn"]["v_proj"].weight
            weight_dict[f"layers.{layer}.attn.output_proj.weight"] = self.layers[layer]["attn"]["output_proj"].weight
            weight_dict[f"layers.{layer}.ffn.w1.weight"] = self.layers[layer]["ffn"]["w1"].weight
            weight_dict[f"layers.{layer}.ffn.w2.weight"] = self.layers[layer]["ffn"]["w2"].weight
            weight_dict[f"layers.{layer}.ln1.weight"] = self.layers[layer]["ln1"].weight
            weight_dict[f"layers.{layer}.ln2.weight"] = self.layers[layer]["ln2"].weight

        # Final layer normalization and output projection
        weight_dict["ln_final.weight"] = self.ln_final.weight
        weight_dict["lm_head.weight"] = self.lm_head.weight

        return weight_dict