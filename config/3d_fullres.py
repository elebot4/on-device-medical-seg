# 3D Full Volume Training - High Accuracy
# For research/server deployment where accuracy matters most

out_dir = 'out-3d-full'
input_shape = (128, 128, 128)  # full 3D volumes

# High-accuracy model
num_stages = 5  # deeper for better features 
base_chs = 64   # more channels
deep_supervision = True
dropout = 0.2   # regularization for better generalization

# Conservative training  
batch_size = 1  # memory intensive with 3D
learning_rate = 3e-4  # conservative lr
weight_decay = 1e-2
nb_epochs = 200  # longer training

# Scheduler for long training
scheduler = 'OneCycleLR' 
gamma = 0.95

device = 'cuda'
dtype = 'float32'  # full precision for research