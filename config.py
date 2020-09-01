import numpy as np


class Config(object):
    NAME = None
    BATCH_SIZE = 8
    MAX_QUEUE_LENGTH = 1024

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    BACKBONE = 'resnet101'
    TRAIN_BN = True
    WEIGHT_DECAY = 1e-4

    # Projection Layers
    NUM_HIDDENS = [256, 128, 50]

    # Cosine Similarity
    TEMPERATURE = 1e-6


class TrainingConfig(Config):
    MODE = "training"

    # Hyperparam of optimizer
    OPTIMIZER = "Ranger"
    LOOKAHEAD = True


class InferenceConfig(Config):
    MODE = "inference"
