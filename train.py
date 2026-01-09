import torch
import torch.optim as optim
import time
import os
from model import HierarchicalReasoningModel
from data_loader import get_loaders
from huggingface_hub import login
login(token="fill") # needed for huggingface datasets
from torch.amp import autocast, GradScaler

def train():
    # Memory Fragmentation Fix
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # --- Auto-Detect Device & Config ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device Detected: {device.upper()}")
    
    if device == "cuda":
        # GPU Config (Colab T4/A100) - "The Beast Mode"
        print(">> GPU Mode Activated: THE BEAST CONFIG")
        # Gradient Checkpointing allows massive batch/depth
        # Reduced Batch to 32 to fix OOM (3.06GB alloc failed)
        BATCH_SIZE = 32   
        BLOCK_SIZE = 256  
        D_MODEL = 768     
        NUM_HEADS = 12    
        NUM_LAYERS = 12   
        GRAD_ACCUM_STEPS = 2 # Effective Batch = 64 (32 * 2)
        
        # Enable Mixed Precision
        use_amp = True
        
        # [CRITICAL UPDATE]
        # Torch.compile (JIT) struggles with RECURSIVE loops + Checkpointing.
        # It caused the "triton_red_fused" crash you saw.
        # We disable it to ensure stability. Flash Attn + TF32 is already fast enough.
        use_compile = False 
    else:
        # CPU Config (Local) - "Lite Mode"
        print(">> CPU Mode Activated: Efficient Config")
        BATCH_SIZE = 4
        BLOCK_SIZE = 128
        D_MODEL = 256
        NUM_HEADS = 4
        NUM_LAYERS = 4
        GRAD_ACCUM_STEPS = 4 
        
        use_amp = False
        use_compile = False

    LEARNING_RATE = 3e-4
    MAX_TOKENS = 500_000_000 
    PATIENCE = 20 
    EVAL_INTERVAL = 100 if device == "cuda" else 50
    EVAL_STEPS = 20 

    print(f"Config: B={BATCH_SIZE}, T={BLOCK_SIZE}, D={D_MODEL}, L={NUM_LAYERS}, Accum={GRAD_ACCUM_STEPS}, AMP={use_amp}")
    
    # ... (Loaders code) ...
    # Prepare Data
    train_loader, val_loader, vocab_size = get_loaders(
        batch_size=BATCH_SIZE, 
        block_size=BLOCK_SIZE, 
        max_tokens_train=MAX_TOKENS,
        max_tokens_val=200000 
    )
    
    val_iter = iter(val_loader) 
    
    # Init Model
    model = HierarchicalReasoningModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_MODEL * 4,
        num_layers=NUM_LAYERS,
        dropout=0.1,
        # rop_embedding handled internally now, just block size logic if needed
    ).to(device)
    
    # Activate Advanced Features
    # Activate Advanced Features
    if device == "cuda":
        # Enable TF32 (huge speedup on Ampere+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(">> TF32 & CuDNN Benchmark ENABLED (Speedup)")
        
        model.use_checkpointing = True
        print(">> Gradient Checkpointing ENABLED (VRAM Saved)")
        
        if use_compile:
            print(">> Compiling Model (torch.compile)...")
            try:
                model = torch.compile(model)
                print(">> Model Compiled! (Expect start-up delay, then 2x speed)")
            except Exception as e:
                print(f"Compilation skipped (Not supported on this env): {e}")

    # Optimizer with Weight Decay (Brain Pruning)
    # Fused AdamW is faster on CUDA
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1, fused=(device=="cuda"))
    
    scaler = GradScaler(enabled=use_amp) 
    
    # Scheduler: Linear Warmup + Cosine Decay
    # OneCycleLR is perfect for this "cool down" strategy
    total_steps = (MAX_TOKENS // (BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM_STEPS)) + 100
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        total_steps=total_steps, 
        pct_start=0.1, # 10% Warmup
        anneal_strategy='cos'
    )
    
    # --- Loop ---
    model.train()
    step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    start_time = time.time()
    print("Optimization Loop Started (with Scheduler & Decay)")
    
    try:
        current_iter_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to Device
            batch = batch.to(device)
            targets = batch.clone()
            
            # Forward (with AMP)
            with autocast(device_type=device, enabled=use_amp):
                logits, loss = model(batch, targets)
                # Scale Loss (Accumulation)
                loss = loss / GRAD_ACCUM_STEPS
            
            # Backward (with Scaler)
            scaler.scale(loss).backward()
            
            current_iter_loss += loss.item()

            # Update Step
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                # Unscale logic for clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step() # Decay LR
                
                step += 1
                
                # Logging (Rescale loss back for display)
                loss_display = current_iter_loss 
                current_iter_loss = 0.0
                
                if step % 10 == 0:
                    dt = time.time() - start_time
                    tps = (step * BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM_STEPS) / dt
                    print(f"Step {step} | Train Loss: {loss_display:.4f} | Tok/sec: {tps:.2f}")

                # VALIDATION STEP
                if step % EVAL_INTERVAL == 0:
                    print("Running Validation...")
                    model.eval()
                    val_losses = []
                    try:
                         with torch.no_grad():
                            for _ in range(EVAL_STEPS):
                                try:
                                    val_batch = next(val_iter)
                                except StopIteration:
                                    print("Validation Iterator Exhausted. Restarting...")
                                    val_iter = iter(val_loader)
                                    val_batch = next(val_iter)
                                
                                val_batch = val_batch.to(device)
                                val_targets = val_batch.clone()
                                
                                with autocast(device_type=device, enabled=use_amp):
                                    _, v_loss = model(val_batch, val_targets)
                                    
                                val_losses.append(v_loss.item())
                         
                         avg_val_loss = sum(val_losses) / len(val_losses)
                         print(f"--> Val Loss: {avg_val_loss:.4f} (Best: {best_val_loss:.4f})")
                         
                         if avg_val_loss < best_val_loss:
                             best_val_loss = avg_val_loss
                             patience_counter = 0
                             torch.save(model.state_dict(), "best_model.pth") 
                             print("Saved Best Model.")
                             
                             # TARGET GOAL CHECK
                             if best_val_loss <= 2.1:
                                 print(f"\nTARGET REACHED! Loss is {best_val_loss:.4f} <= 2.1")
                                 print("Stopping training as 'Full Knowledge' has been achieved.")
                                 break
                         else:
                             patience_counter += 1
                             print(f"No improvement. Patience {patience_counter}/{PATIENCE}")
                             if patience_counter >= PATIENCE:
                                 print(f"Early Stopping! Model started overfitting (Val Loss rose).")
                                 break
                    except Exception as e:
                        print(f"Validation Failed (Skipping this check): {e}")
                    
                    model.train() # Back to train mode

    except KeyboardInterrupt:
        print("Training interrupted manually.")
    
    print(f"Training Complete. Best Val Loss: {best_val_loss}")
    if os.path.exists("best_model.pth"):
        print("Best model saved to 'best_model.pth'")

if __name__ == "__main__":
    train()
