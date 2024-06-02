import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNet

def solution_v1():
    # Load the pre-trained MobileNet model without the top classification layer
    pre_trained_model = MobileNet(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

    # Freeze all the layers in the pre-trained model initially
    for layer in pre_trained_model.layers:
        layer.trainable = False

    # Add custom layers on top of the pre-trained model
    last_layer = pre_trained_model.output

    train_dir = 'skin2_split/train'
    validation_dir = 'skin2_split/val'
    test_dir = 'skin2_split/test'

    def resize_image(image):
        return tf.image.resize(image, (150, 150))

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        zoom_range=0.2,
        shear_range=0.2,
        height_shift_range=0.2,
        width_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.2, 1.0],
        fill_mode='nearest',
        rescale=1.0 / 255,
        preprocessing_function=resize_image
    )

    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255.,
        preprocessing_function=resize_image
    )

    test_datagen = validation_datagen

    train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        target_size=(150, 150))

    validation_generator = validation_datagen.flow_from_directory(directory=validation_dir,
                                                                  batch_size=32,
                                                                  class_mode='categorical',
                                                                  target_size=(150, 150))

    test_generator = test_datagen.flow_from_directory(directory=test_dir,
                                                      batch_size=32,
                                                      class_mode='categorical',
                                                      target_size=(150, 150))

    x = layers.GlobalAveragePooling2D()(last_layer)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    num_classes = train_generator.num_classes
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(pre_trained_model.input, x)

    # Unfreeze the last few layers of the base model
    for layer in pre_trained_model.layers[-20:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_generator,
                        epochs=50,
                        verbose=1,
                        validation_data=validation_generator)

    test_loss, test_acc = model.evaluate(test_generator)
    print('Test accuracy:', test_acc)

    return model

if __name__ == '__main__':
    model = solution_v1()
    model.save("model_mobilenet_v2.h5")
