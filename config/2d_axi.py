# 2D Axial slice training - mobile optimized
# Small model for fast inference on edge devices

out_dir = 'out-2d-axi' 
input_shape = (256, 256)  # 2D axial slices
in_channels = 4  # multi-modal input

# Mobile-optimized model  
num_stages = 3  # smaller for speed
base_chs = 16   # fewer channels
dropout = 0.0   # no dropout for inference speed

# Fast training
batch_size = 8  # can use larger batch with 2D
learning_rate = 1e-3  # can go higher with smaller model
nb_epochs = 50  # fewer epochs needed

# For mobile deployment
dtype = 'float16'  # mixed precision for speed