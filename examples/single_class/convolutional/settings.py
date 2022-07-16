
import matplotlib.pyplot as plt

# plot settings
plt.rc("figure", dpi=200)


# pytorch settings
DEVICE = "cuda"

# train settings
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
GRADIENT_CLIP = 0.5
MAX_EPOCHS = 500
N_LAYERS = 12

# data settings
N_TRAIN = 750
N_VAL = 100

IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_SHAPE = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
BATCH_SHAPE = (BATCH_SIZE, *IMAGE_SHAPE)
