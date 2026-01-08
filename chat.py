import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import HierarchicalReasoningModel
import os

def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.6, top_k=50, top_p=0.9, device='cpu'):
    # Encode
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Context Window Constraint
            cond_ids = input_ids[:, -128:] 
            
            logits, _ = model(cond_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            # --- 1. N-Gram Blocking (Manual) ---
            # If we produced "A B A", prevent "B" from happening again if it forms 3-gram
            # Simplified: Just ban the immediate previous Bigram from repeating instantly? 
            # Or strict "No Repeat N-Gram Size = 3"
            # Hard to do efficiently in Python loop without cache, 
            # so we'll do the "Symbol Spam" blocker:
            # If last 3 tokens are same, BAN that token.
            if input_ids.size(1) > 3:
                last_tokens = input_ids[0, -3:].tolist()
                if last_tokens[0] == last_tokens[1] == last_tokens[2]:
                    # Ban this token
                    next_token_logits[0, last_tokens[2]] = float('-inf')

            # --- 2. Repetition Penalty ---
            for i in range(input_ids.size(1)):
                token_id = input_ids[0, i].item()
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= 1.2 
                else:
                    next_token_logits[0, token_id] *= 1.2 

            # --- 3. Top-K Filtering ---
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')

            # --- 4. Top-P (Nucleus) Filtering ---
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[0, indices_to_remove] = float('-inf')

            probs = F.softmax(next_token_logits, dim=-1)
            
            # Safety check for NaNs if we filtered everything (unlikely but possible)
            if torch.isnan(probs).any():
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def chat():
    print("Loading Model...")
    
    # Config (Must match train.py)
    D_MODEL = 256
    NUM_HEADS = 4
    NUM_LAYERS = 4
    BLOCK_SIZE = 128
    
    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except:
        print("Using Mock Tokenizer (No real generation possible).")
        # Mock for offline testing logic
        class MockTok:
             vocab_size = 50257
             eos_token_id = 50256
             def encode(self, text, **kwargs): return torch.tensor([[1, 2, 3]])
             def decode(self, ids, **kwargs): return " [Mock Output] "
        tokenizer = MockTok()

    # Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device.upper()}")
    
    # Locate Model
    model_path = None
    if os.path.exists("best_model.pth"):
        model_path = "best_model.pth"
    elif os.path.exists("best_model_cpu.pth"):
        model_path = "best_model_cpu.pth"
    else:
        print("No model found! (best_model.pth or best_model_cpu.pth)")
        return

    print(f"Loading weights from {model_path}...")
    
    # Inspect Checkpoint for Config
    try:
        # Load dictionary (map_location handles CPU/GPU mismatch)
        state_dict = torch.load(model_path, map_location=device)
        
        # Deduce D_MODEL from embedding weight shape: [vocab_size, d_model]
        if "token_embedding.weight" in state_dict:
            detected_d_model = state_dict["token_embedding.weight"].shape[1]
            print(f"Auto-Detected D_MODEL: {detected_d_model}")
        else:
            detected_d_model = 512 if "cuda" in device else 256
            
        # Deduce BLOCK_SIZE (max_len) from position embedding: [max_len, d_model]
        if "position_embedding.weight" in state_dict:
            detected_max_len = state_dict["position_embedding.weight"].shape[0]
            print(f"Auto-Detected MAX_LEN: {detected_max_len}")
        else:
            detected_max_len = 256 # Fallback
            
        # Set Config based on D_MODEL
        D_MODEL = detected_d_model
        MAX_LEN = detected_max_len
        
        if D_MODEL == 512:
            NUM_HEADS = 8
            NUM_LAYERS = 8
        elif D_MODEL == 256:
            NUM_HEADS = 4
            NUM_LAYERS = 4
        elif D_MODEL == 768: 
            NUM_HEADS = 12
            NUM_LAYERS = 12
        else:
            NUM_HEADS = 8
            NUM_LAYERS = 8
            
    except Exception as e:
        print(f"Error inspecting checkpoint: {e}")
        return

    print(f"Initializing Model (D={D_MODEL}, H={NUM_HEADS}, L={NUM_LAYERS}, T={MAX_LEN})...")
    model = HierarchicalReasoningModel(
        vocab_size=50257, 
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_MODEL*4,
        num_layers=NUM_LAYERS,
        dropout=0.1,
        max_len=MAX_LEN
    ).to(device)
    
    try:
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        return

    model.eval()
    
    print("\n--- Start Chatting (type 'quit' to exit) ---")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        response = generate(model, tokenizer, user_input, device=device)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
