import torch
import numpy as np
import os
import sys

def save_flat_binary(tensor, filepath):
    """Converts a PyTorch tensor to raw float32 bytes and saves it."""
    # Detach from graph, convert to float32 (in case of fp16), and flatten
    flat_array = tensor.detach().cpu().float().numpy().astype(np.float32).flatten()
    raw_bytes = flat_array.tobytes()
    
    with open(filepath, "wb") as f:
        f.write(raw_bytes)
        
    size_mb = len(raw_bytes) / (1024 * 1024)
    print(f" -> Saved {os.path.basename(filepath)} ({size_mb:.2f} MB)")

def export_cluster_weights(model_path, output_dir, num_compute_nodes=6):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading PyTorch model from {model_path}...\n")
    
    # Load the model weights into RAM
    # TinyStories models are usually saved directly as state_dicts
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Some checkpoints wrap the state_dict inside a 'model' key
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # ========================================================
    # NODE 1: The Head (Token Embeddings)
    # ========================================================
    print("--- Extracting Node 1 (Head) ---")
    if 'tok_embeddings.weight' in state_dict:
        save_flat_binary(state_dict['tok_embeddings.weight'], os.path.join(output_dir, "node1_embeddings.bin"))
    else:
        print(" [!] Could not find token embeddings!")

    # ========================================================
    # NODES 2-7: The Compute Chain (Transformer Layers)
    # ========================================================
    print("\n--- Extracting Nodes 2-7 (Compute Chain) ---")
    
    # Standard Llama weight suffixes per layer
    layer_components = [
        "attention_norm.weight",
        "attention.wq.weight",
        "attention.wk.weight",
        "attention.wv.weight",
        "attention.wo.weight",
        "ffn_norm.weight",
        "feed_forward.w1.weight",
        "feed_forward.w2.weight",
        "feed_forward.w3.weight"
    ]

    for layer_idx in range(num_compute_nodes):
        node_id = layer_idx + 2
        print(f"Slicing Layer {layer_idx} for Node {node_id}...")
        
        layer_prefix = f"layers.{layer_idx}."
        concatenated_arrays = []
        
        for component in layer_components:
            key = layer_prefix + component
            if key in state_dict:
                # Convert to raw 32-bit float array and flatten
                arr = state_dict[key].detach().cpu().float().numpy().astype(np.float32).flatten()
                concatenated_arrays.append(arr)
            else:
                print(f" [!] Missing expected weight: {key}")
                
        if concatenated_arrays:
            # Merge all matrices for this layer into one massive, sequential 1D array
            # This allows the ESP32 to read them linearly out of the SPI Flash
            node_block = np.concatenate(concatenated_arrays)
            filepath = os.path.join(output_dir, f"node{node_id}_layer{layer_idx}.bin")
            
            with open(filepath, "wb") as f:
                f.write(node_block.tobytes())
            
            size_mb = len(node_block.tobytes()) / (1024 * 1024)
            print(f" -> Saved {os.path.basename(filepath)} ({size_mb:.2f} MB)")

    # ========================================================
    # NODE 8: The Tail (Classifier)
    # ========================================================
    print("\n--- Extracting Node 8 (Tail) ---")
    
    # Extract the final LayerNorm
    if 'norm.weight' in state_dict:
        save_flat_binary(state_dict['norm.weight'], os.path.join(output_dir, "node8_final_norm.bin"))
    
    # Extract Output Classifier (handling weight tying)
    if 'output.weight' in state_dict:
        save_flat_binary(state_dict['output.weight'], os.path.join(output_dir, "node8_classifier.bin"))
    elif 'tok_embeddings.weight' in state_dict:
        print(" -> Weight tying detected. Cloning embeddings for classifier.")
        save_flat_binary(state_dict['tok_embeddings.weight'], os.path.join(output_dir, "node8_classifier.bin"))

    print("\nExtraction Complete! Files are ready to be flashed to the ESP32s.")

if __name__ == "__main__":
    # Change this to the path of your downloaded PyTorch model
    MODEL_FILE = "tinystories_15m.pt" 
    OUTPUT_FOLDER = "./cluster_binaries"
    
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file '{MODEL_FILE}' not found.")
        sys.exit(1)
        
    export_cluster_weights(MODEL_FILE, OUTPUT_FOLDER)
