# 2D Axial slice training - mobile optimized
# Fast training on axial slices for efficient mobile deployment

# I/O settings
out_dir = 'checkpoints/2d_axi'

# Data settings  
input_shape = (256, 256)  # 2D axial slices
batch_size = 8
slice_mode = 'axi'  # axial slices

# Model architecture (smaller for mobile)
num_stages = 3  # Fewer stages for mobile
base_chs = 24   # Smaller base channels
dropout = 0.1

# Training settings
nb_epochs = 50  # Faster training
learning_rate = 1e-3
weight_decay = 5e-3

# Mixed precision for efficiency
dtype = 'float16'
