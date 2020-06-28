import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.utils as KU
from tensorflow.keras.layers import Layer


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


def load_image_gt(dataset, image_id, augmentation=None):
    image = dataset.load_image(image_id)
    images = np.array([image, image])
    if augmentation:
        images = augmentation(images=images)
    return images


class DataGenerator(KU.Sequence):
    def __init__(self, dataset, config, shuffle=True, augmentation=None):
        self.image_ids = np.copy(dataset.image_ids)
        self.dataset = dataset
        self.config = config

        self.shuffle = shuffle
        self.augmentation = augmentation
        self.batch_size = self.config.BATCH_SIZE

        if self.shuffle is True:
            np.random.shuffle(self.image_ids)

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size))) - 1

    def __getitem__(self, idx):
        return self.data_generator(
            self.image_ids[idx*self.batch_size:(idx+1)*self.batch_size])

    def data_generator(self, image_ids):
        b = 0
        while b < self.batch_size and b < image_ids.shape[0]:
            image_id = image_ids[b]
            paired_images = load_image_gt(
                self.dataset, image_id, self.augmentation)

            if b == 0:
                batch_images = np.zeros(
                    (2*self.batch_size,) + paired_images.shape[1:],
                    dtype=np.float32)
                batch_labels = np.zeros(
                    (2*self.batch_size)
                )

            batch_images[2*b:2*b+2] = paired_images
            batch_labels[2*b:2*b+2] = np.array([2*b+1, 2*b])
            b += 1

            if b >= self.batch_size:
                inputs = [batch_images]
                outputs = [batch_labels]
                return inputs, outputs

    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.image_ids)


class CosineSimilarity(Layer):
    def __init__(self, batch_size):
        super(CosineSimilarity, self).__init__()
        large_num = 1e9
        self.mask = -large_num*tf.eye(2*batch_size)

    def call(self, z):
        z = tf.math.l2_normalize(z, axis=1)
        z = K.dot(z, K.transpose(z))
        z = z + self.mask
        return z


class SimCLRv2(object):
    def __init__(self, config):
        self.config = config
        self.build()

    def build(self):
        config = self.config

        # Inputs
        input_image = KL.Input(
            shape=[None, None, 3], name="input_image")

        # Build Resnet
        _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
                                         stage5=True, train_bn=config.TRAIN_BN)

        z2 = KL.GlobalAveragePooling2D()(C2)
        z3 = KL.GlobalAveragePooling2D()(C3)
        z4 = KL.GlobalAveragePooling2D()(C4)
        z5 = KL.GlobalAveragePooling2D()(C5)

        z = KL.Concatenate(axis=-1)([z2, z3, z4, z5])

        logits = CosineSimilarity(config.BATCH_SIZE)(z)
        probs = KL.Activation('softmax')(logits)
        self.model = KM.Model(input_image, probs)

    def compile(self):
        self.model.compile(
            optimizer="Adam",
            loss='sparse_categorical_crossentropy',
        )

    def save_h5(self):
        pass

    def train(self, dataset, augmentation):
        train_generator = DataGenerator(
            dataset, self.config, augmentation=augmentation
            )
        self.model.fit(train_generator)
