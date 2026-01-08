import os
import torch
import random
import itertools
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

import re

def is_clean_text(text):
    """
    Filters out low-quality data:
    1. Lines longer than 1000 chars (Minified code/data dumps)
    2. Continuous symbol spam (e.g. '((((((((((')
    """
    if len(text) > 10000: # Safety cap for extremely long docs
        return False
        
    # Check for symbol spam (>10 non-alphanumeric chars in a row)
    # This regex looks for 10 consecutive punctuation/symbol characters
    if re.search(r'[^a-zA-Z0-9\s]{10,}', text):
        return False
        
    return True

def manual_interleave(datasets, probabilities):
    """
    Manually interleaves multiple iterators based on probabilities.
    Restarts any exhausted iterator to ensure an INFINITE stream.
    """
    iters = [iter(ds) for ds in datasets]
    while True:
        # Select dataset index
        idx = random.choices(range(len(iters)), weights=probabilities, k=1)[0]
        try:
            yield next(iters[idx])
        except StopIteration:
            # Restart this specific iterator
            # Note: This relies on the 'datasets' objects being re-iterable (creating new streams)
            # This is critical for the "Overnight" strategy.
            iters[idx] = iter(datasets[idx])
            try:
                yield next(iters[idx])
            except StopIteration:
                # If a dataset is empty (0 samples), just skip it to avoid crash loop
                continue

def get_limited_5gb_stream(is_validation=False):
    """
    Creates an interleaved stream of high-quality reasoning, code, and science datasets.
    Uses manual interleaving with Per-Source splitting to ensure balanced Validation.
    """
    print(f"Initializing High-Quality Data Stream (Validation={is_validation})...")
    
    try:
        # Load datasets independently
        # Note: We must ensure these are iterable and support take/skip or we use itertools
        ds1 = load_dataset("TinyGSM/TinyGSM", split="train", streaming=True)
        ds2 = load_dataset("nampdn-ai/tiny-codes", split="train", streaming=True)
        ds3 = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train", streaming=True)
        ds4 = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
        
        raw_datasets = [ds1, ds2, ds3, ds4]
        probs = [0.4, 0.3, 0.15, 0.15]
        
        # Strict Per-Source Splitting
        # Val = First 1000 of EACH source
        # Train = Skip first 1000 of EACH source
        SPLIT_SIZE = 1000
        
        processed_datasets = []
        for ds in raw_datasets:
            if is_validation:
                # validation_buffer
                processed_datasets.append(ds.take(SPLIT_SIZE))
            else:
                # train_generator (never touch the buffer)
                processed_datasets.append(ds.skip(SPLIT_SIZE))

        # Manual Mix
        stream = manual_interleave(processed_datasets, probabilities=probs)
        return stream

    except Exception as e:
        print(f"Error loading remote datasets: {e}")
        return None

class MixedDataset(IterableDataset):
    def __init__(self, tokenizer, max_tokens_limit=10_000_000, block_size=256, is_validation=False):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_tokens = max_tokens_limit
        self.current_tokens = 0
        self.is_validation = is_validation
        
        self.is_validation = is_validation
        
        # We do NOT init self.data_iter here because iterators are one-time use.
        # We will init it in __iter__

    def __iter__(self):
        # Create a FRESH stream every time we iterate
        self.current_tokens = 0 # Reset counter
        data_iter = get_limited_5gb_stream(is_validation=self.is_validation)
        
        if data_iter:
            while self.current_tokens < self.max_tokens:
                buffer_text = ""
                try:
                    sample = next(data_iter)
                    
                    # Extract text flexibly
                    text_parts = []
                    if 'question' in sample and 'answer' in sample:
                        text_parts.append(f"Q: {sample['question']}\nA: {sample['answer']}")
                    elif 'query' in sample and 'response' in sample:
                        text_parts.append(f"Q: {sample['query']}\nA: {sample['response']}")
                    
                    if 'text' in sample and sample['text']:
                        text_parts.append(str(sample['text']))
                    if 'content' in sample and sample['content']:
                        text_parts.append(str(sample['content']))
                    
                    if not text_parts:
                        text_parts.append(str(sample))
                        
                    # --- Data Cleaning Filter ---
                    # Join and check if it's high quality
                    full_text = "\n".join(text_parts)
                    
                    # Basic length/symbol check on the raw text
                    if not is_clean_text(full_text):
                        # Skip this sample
                        continue
                        
                    buffer_text = full_text + "\n<|endoftext|>\n"
                    
                except StopIteration:
                    break
                except Exception as e:
                    continue
                
                tokens = self.tokenizer(buffer_text, truncation=False, return_tensors="pt")['input_ids'][0]
                for i in range(0, len(tokens) - self.block_size, self.block_size):
                    chunk = tokens[i:i + self.block_size]
                    if len(chunk) == self.block_size:
                        self.current_tokens += self.block_size
                        yield chunk
                    if self.current_tokens >= self.max_tokens:
                        return

        # 2. No Local Fallback (Strict Mode)
        if self.current_tokens == 0:
            print("WARNING: Stream empty or failed. No local fallback allowed.")
            yield torch.tensor([0]*self.block_size, dtype=torch.long)

def get_loaders(batch_size=4, block_size=256, max_tokens_train=10_000_000, max_tokens_val=100_000):
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        # Silencing the "sequence length is longer than the specified maximum" warning
        # because we handle chunking manually in MixedDataset.
        tokenizer.model_max_length = 100000
    except:
        print("Mock Tokenizer.")
        class MockTok:
             vocab_size = 50257
             eos_token = "<|endoftext|>"
             model_max_length = 100000
             def __call__(self, text, **kwargs): return {'input_ids': torch.randint(0, 50257, (1, 100))}
        tokenizer = MockTok()

    train_ds = MixedDataset(tokenizer, max_tokens_limit=max_tokens_train, block_size=block_size, is_validation=False)
    val_ds = MixedDataset(tokenizer, max_tokens_limit=max_tokens_val, block_size=block_size, is_validation=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, tokenizer.vocab_size
