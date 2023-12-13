from keras import Input
from keras.src.layers import Activation, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D
from numpy import concatenate
from pycocotools.coco import COCO
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from src.preprocessing.preprocess import CustomDataGenerator, filter_images, generate_masks
from src.utility.path_utils import get_path_from_root

# Constants and Path Initialization
ANNOTATION_FILE_TRAIN = get_path_from_root("data", "coco", "annotations", "train&val", "instances_train2014.json")
IMAGE_DIR = get_path_from_root("data", "coco", "images", "train")
MASK_DIR = get_path_from_root("data", "mask", "train")
TARGET_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 20

# Initialize COCO API
coco = COCO(ANNOTATION_FILE_TRAIN)


def down_block(
        input_tensor,
        no_filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_normal",
        max_pool_window=(2, 2),
        max_pool_stride=(2, 2)
):
    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(input_tensor)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    # conv for skip connection
    conv = Activation("relu")(conv)

    pool = MaxPooling2D(pool_size=max_pool_window, strides=max_pool_stride)(conv)

    return conv, pool


def bottle_neck(
        input_tensor,
        no_filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_normal"
):
    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(input_tensor)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    return conv


def up_block(
        input_tensor,
        no_filters,
        skip_connection,
        kernel_size=(3, 3),
        strides=(1, 1),
        upsampling_factor=(2, 2),
        max_pool_window=(2, 2),
        padding="same",
        kernel_initializer="he_normal"
):
    conv = Conv2D(
        filters=no_filters,
        kernel_size=max_pool_window,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(UpSampling2D(size=upsampling_factor)(input_tensor))

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    conv = concatenate([skip_connection, conv], axis=-1)

    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    return conv


def output_block(
        input_tensor,
        padding="same",
        kernel_initializer="he_normal"
        ):
    conv = Conv2D(
        filters=2,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation="relu",
        padding=padding,
        kernel_initializer=kernel_initializer
    )(input_tensor)

    conv = Conv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation="sigmoid",
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    return conv


# UNet Model
def UNet(input_shape=(128, 128, 3)):
    filter_size = [64, 128, 256, 512, 1024]

    inputs = Input(shape=input_shape)

    d1, p1 = down_block(input_tensor=inputs,
                        no_filters=filter_size[0],
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding="same",
                        kernel_initializer="he_normal",
                        max_pool_window=(2, 2),
                        max_pool_stride=(2, 2))

    d2, p2 = down_block(input_tensor=p1,
                        no_filters=filter_size[1],
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding="same",
                        kernel_initializer="he_normal",
                        max_pool_window=(2, 2),
                        max_pool_stride=(2, 2))

    d3, p3 = down_block(input_tensor=p2,
                        no_filters=filter_size[2],
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding="same",
                        kernel_initializer="he_normal",
                        max_pool_window=(2, 2),
                        max_pool_stride=(2, 2))

    d4, p4 = down_block(input_tensor=p3,
                        no_filters=filter_size[3],
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding="same",
                        kernel_initializer="he_normal",
                        max_pool_window=(2, 2),
                        max_pool_stride=(2, 2))

    b = bottle_neck(input_tensor=p4,
                    no_filters=filter_size[4],
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer="he_normal")

    u4 = up_block(input_tensor=b,
                  no_filters=filter_size[3],
                  skip_connection=d4,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor=(2, 2),
                  max_pool_window=(2, 2),
                  padding="same",
                  kernel_initializer="he_normal")

    u3 = up_block(input_tensor=u4,
                  no_filters=filter_size[2],
                  skip_connection=d3,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor=(2, 2),
                  max_pool_window=(2, 2),
                  padding="same",
                  kernel_initializer="he_normal")

    u2 = up_block(input_tensor=u3,
                  no_filters=filter_size[1],
                  skip_connection=d2,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor=(2, 2),
                  max_pool_window=(2, 2),
                  padding="same",
                  kernel_initializer="he_normal")

    u1 = up_block(input_tensor=u2,
                  no_filters=filter_size[0],
                  skip_connection=d1,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor=(2, 2),
                  max_pool_window=(2, 2),
                  padding="same",
                  kernel_initializer="he_normal")

    output = output_block(input_tensor=u1,
                          padding="same",
                          kernel_initializer="he_normal")

    model = keras.models.Model(inputs=inputs, outputs=output)

    return model


# Update preprocess.py to skip missing images
def preprocess_data(coco, image_dir, mask_dir, category_names, target_size):
    imgIds = filter_images(coco, category_names)
    generate_masks(coco, imgIds, mask_dir, image_dir, category_names)


# Training Function
def train_model():
    # Preprocess the data
    preprocess_data(coco, IMAGE_DIR, MASK_DIR, ['person'], TARGET_SIZE)

    # Initialize the data generators
    train_imgIds = filter_images(coco, ['person'])
    train_generator = CustomDataGenerator(coco, train_imgIds, IMAGE_DIR, MASK_DIR, BATCH_SIZE)

    # Create the UNet model
    model = UNet(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the model with the training generator
    train_steps = len(train_imgIds) // BATCH_SIZE
    model.fit(train_generator, steps_per_epoch=train_steps, epochs=EPOCHS)

    # Save the model
    model.save('unet_model.h5')


if __name__ == "__main__":
    train_model()
