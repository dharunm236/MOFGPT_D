import random
import os

def split_mofid_file(input_path, output_dir, seed=42):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the file
    with open(input_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Shuffle for randomness
    random.seed(seed)
    random.shuffle(lines)

    # Compute sizes
    total = len(lines)
    n_train = int(total * 0.80)
    n_val = int(total * 0.05)
    n_test = total - n_train - n_val  # Remainder to test

    # Split
    train_lines = lines[:n_train]
    val_lines = lines[n_train:n_train + n_val]
    test_lines = lines[n_train + n_val:]

    # Write to files
    with open(os.path.join(output_dir, 'train.csv'), 'w') as f:
        f.write('\n'.join(train_lines))
    
    with open(os.path.join(output_dir, 'val.csv'), 'w') as f:
        f.write('\n'.join(val_lines))
    
    with open(os.path.join(output_dir, 'test.csv'), 'w') as f:
        f.write('\n'.join(test_lines))

    print(f"Split complete: {len(train_lines)} train, {len(val_lines)} val, {len(test_lines)} test")

# Example usage
split_mofid_file('QMOF/mofid/QMOF_small_mofid.csv', 'splits/')
