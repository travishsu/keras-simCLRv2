import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import re

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.utils as KU
from tensorflow.keras.layers import Layer

import tensorflow_addons as tfa

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# class BatchNorm(KL.BatchNormalization):
class BatchNorm(tfa.layers.GroupNormalization):
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
        # return super(self.__class__, self).call(inputs, training=training)
        return super(self.__class__, self).call(inputs)


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
    x = BatchNorm(gamma_initializer='zeros', name=bn_name_base + '2c')(x, training=train_bn)

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
    shortcut = BatchNorm(gamma_initializer='zeros', name=bn_name_base + '1')(shortcut, training=train_bn)

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
        images = np.array(augmentation(images=images))
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

            batch_images[2*b:2*b+2] = mold_image(paired_images, self.config)
            batch_labels[2*b:2*b+2] = np.array([2*b+1, 2*b])
            b += 1

            if b >= self.batch_size:
                inputs = [batch_images]
                outputs = [batch_labels]
                return inputs, outputs

    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.image_ids)


class OnEpochEnd(tf.keras.callbacks.Callback):
    """Workaround to deal with bug in Tensorflow 2.1.0

    "tf.keras.fit not calling Sequence.on_epoch_end" refer 
    to tenstflow/issues/#35911
    """
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback()


# TODO: Momentum weight
class MoCoQueue(Layer):
    """This layer provides queue mechanism to save intput as previous projection results,
    and return current the whole queue as negative sample in each iteration.
    """
    def __init__(self, config):
        super(MoCoQueue, self).__init__()
        self.batch_size = config.BATCH_SIZE
        self.max_queue_length = config.MAX_QUEUE_LENGTH
        self.c = (self.max_queue_length+2*self.batch_size) // (2*self.batch_size)

    def build(self, input_shape):
        self.embedding_dim = input_shape[-1]
        init = tf.random_normal_initializer()
        self.keys = tf.Variable(
            initial_value=init(shape=(self.max_queue_length, self.embedding_dim), dtype='float32'),
            trainable=False)
    
    def call(self, new_keys):
        keys = self.keys
        cat = tf.concat([new_keys, self.keys], 0)
        self.keys.assign(
            tf.slice(cat, [0, 0], [self.max_queue_length, self.embedding_dim],))
        return keys


class CosineSimilarity(Layer):
    """Compute in-batch similarity and similarities between current batch and MoCo queue.
    """
    def __init__(self, config):
        super(CosineSimilarity, self).__init__()
        large_num = 1e9
        self.temperature = config.TEMPERATURE

        self.batch_size = config.BATCH_SIZE
        self.max_queue_length = config.MAX_QUEUE_LENGTH
        self.c = (self.max_queue_length) // (2*self.batch_size)

        self.mask = large_num*tf.eye(2*self.batch_size)

    def call(self, z, moco):
        z_moco = K.dot(z, K.transpose(moco))  / self.temperature
        z = K.dot(z, K.transpose(z))  / self.temperature
        z = z - self.mask
        z = tf.concat([z, z_moco], axis=-1)
        return z


class SimCLRv2(object):
    def __init__(self, config):
        self.config = config
        self.build(mode=self.config.MODE)

    def build(self, mode='training'):
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

        z = BatchNorm(name='projection_bn')(z)

        z = KL.Dense(config.NUM_HIDDENS[0], name='projection1')(z)
        z = KL.Activation('relu')(z)
        z = KL.Dense(config.NUM_HIDDENS[1], name='projection2')(z)
        z = KL.Activation('relu')(z)
        z = KL.Dense(config.NUM_HIDDENS[2], name='projection3')(z)

        z = KL.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(z)

        if mode == 'training':
            keys = MoCoQueue(config)(  z)
            logits = CosineSimilarity(config)(z, keys)
            probs = KL.Activation('softmax')(logits)
            output = probs
        else:
            output = z
        self.model = KM.Model(input_image, output)

    def load_h5(self, path):
        self.model.load_weights(path, by_name=True)

    def save_h5(self, path):
        self.model.save_weights(path)

    def compile(self, learning_rate, lookahead=True):
        config = self.config
        # Optimizer object
        if config.OPTIMIZER == "SGD+Momentum":
            optimizer = tf.keras.optimizers.SGD(
                lr=learning_rate, momentum=momentum)
        elif config.OPTIMIZER == "AdamW":
            optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-8)
        elif config.OPTIMIZER == "Ranger":
            optimizer = tfa.optimizers.RectifiedAdam(lr=learning_rate)
        else:
            raise ValueError("Your config.OPTIMIZER is not given or not supported in this repo.")

        if config.LOOKAHEAD:
            optimizer = tfa.optimizers.Lookahead(optimizer)

        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
        )

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            print("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                print("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def train(self, dataset, augmentation, epochs=1, learning_rate=1e-2, layers='all'):

        # Pre-defined layer regular expressions
        layer_regex = {
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
            "5+": r"(res5.*)|(bn5.*))",
            "resnet": r"(res.*)|(bn.*)",
            "projection": r"(projection_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        train_generator = DataGenerator(
            dataset, self.config, augmentation=augmentation
            )
        callbacks = [
            OnEpochEnd([train_generator.on_epoch_end]),
            ]
        self.set_trainable(layers, keras_model=self.model, verbose=0)
        self.compile(learning_rate=learning_rate)
        self.model.fit(
            train_generator,
            epochs=epochs,
            use_multiprocessing=True,
            callbacks=callbacks,
            )


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL
