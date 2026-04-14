# Example usage of the new nanoGPT-style configuration

# Set some test variables
input_shape = (32, 32, 32)
in_channels = 1
out_channels = 2  
num_stages = 3
base_chs = 16
norm_type = 'group'
act_type = 'relu'
dropout = 0.0
norm_groups = 8
deep_supervision = True

print(f"Test config: {num_stages} stages, {base_chs} base channels")
print(f"Input: {input_shape}, channels: {in_channels} -> {out_channels}")
print("Variables-based configuration is ready to use!")