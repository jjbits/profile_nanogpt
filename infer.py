import argparse
import torch
from torch.nn import functional as F
import tiktoken

from train_gpt2 import GPT, GPTConfig

def generate(model, prompt, max_length=100, top_k=50, device="cuda"):
    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        while tokens.size(1) < max_length:
            with torch.autocast(device_type=device if device != "cpu" else "cpu", dtype=torch.bfloat16):
                logits, _ = model(tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            next_token = torch.gather(topk_indices, -1, ix)
            tokens = torch.cat((tokens, next_token), dim=1)

    return enc.decode(tokens[0].tolist())

def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--prompt", type=str, default="Hello, I'm a language model,", help="Prompt to start generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Create model from saved config
    config = checkpoint['config']
    model = GPT(config)
    model = torch.compile(model)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    print(f"Model loaded (step {checkpoint['step']}, val_loss {checkpoint['val_loss']:.4f})")
    print(f"Prompt: {args.prompt}")
    print("-" * 50)

    # Generate
    output = generate(model, args.prompt, args.max_length, args.top_k, device)
    print(output)

if __name__ == "__main__":
    main()
