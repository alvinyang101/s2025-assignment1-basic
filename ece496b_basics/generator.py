import torch
from ece496b_basics.tokenizer import Tokenizer
from ece496b_basics.transformers import softmax

def generate_text(model, tokenizer, prompt, max_tokens=50, temperature=1.0, top_p=0.9, device='cpu'):
    """
    Generate text from the model using temperature scaling and nucleus (top-p) sampling.
    
    Args:
        model: Trained transformer language model.
        tokenizer (Tokenizer): Tokenizer for encoding and decoding text.
        prompt (str): Input text prompt.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Softmax temperature for randomness control.
        top_p (float): Probability threshold for nucleus sampling.
        device (str): Device to run the model on ('cpu' or 'cuda').
    
    Returns:
        str: Generated text.
    """
    model.to(device)
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    generated = input_ids[:]
    
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(input_tensor)[:, -1, :]
            logits = logits / temperature
            probs = softmax(logits, dim=-1)
            
            # Nucleus (top-p) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs > top_p
            top_p_mask[:, 1:] = top_p_mask[:, :-1].clone()
            top_p_mask[:, 0] = False
            sorted_probs[top_p_mask] = 0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            
            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token)
            next_token_id = next_token.item()
            
            # Append generated token
            generated.append(next_token_id)
            
            next_token = next_token.view(1, 1)
            input_tensor = torch.cat([input_tensor, next_token], dim=1) 
            context_length = model.context_length
            if input_tensor.shape[1] >= context_length:
                input_tensor = input_tensor[:, -context_length:]
    
    return tokenizer.decode(generated)