import keras
from keras.datasets import cifar10
from keras.layers import *
from keras.models import Model, Input
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import os
from math import ceil

# bottle_neck resnet module consists of 3 conv layers, 1*1conv, 3*3conv, 1*1conv 
def resnet_module(x, filters, pool = False):
    res = x
    stride = 1
    if pool:
        stride = 2
        res = Conv2D(filters, kernel_size = 1, strides = 2, padding = "same")(res)
        res = BatchNormalization()(res)
        

    x = Conv2D(int(filters/4), kernel_size = 1, strides = 1,  padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(int(filters/4), kernel_size = 3, strides = stride,  padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters, kernel_size = 1, strides = 1,  padding = "same")(x)
    x = BatchNormalization()(x)
    
    x = add([x, res])
    x = Activation("relu")(x)

    return x

def resnet_first_module(x, filters):
    res = x
    stride = 1
    res = Conv2D(filters, kernel_size = 1, strides = 1, padding = "same")(res)
    res = BatchNormalization()(res)


    x = Conv2D(int(filters/4), kernel_size = 1, strides = 1,  padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(int(filters/4), kernel_size = 3, strides = stride,  padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, kernel_size = 1, strides = 1,  padding = "same")(x)
    x = BatchNormalization()(x)
    
    x = add([x, res])
    x = Activation("relu")(x)

    return x

def resnet_block(x, filters, num_layers, pool_first_layer = True):
    for a in range(num_layers):
        pool = False
        if a == 0 and pool_first_layer:pool = True
        x = resnet_module(x, filters = filters, pool = pool)
    return x        


def resnet_model(input_shape, num_layers = 50, num_classes = 10):
    if num_layers not in [101, 50, 152]:
        raise ValueError("Num_layers does not exists")

    block_layers = {50: [3, 4, 6, 3],
                    101: [3, 4, 23, 3],
                    152: [3, 8, 36, 3]}   

    block_filters = {50: [256, 512, 1024, 2048],
                     101: [256, 512, 1024, 2048],
                     152: [256, 512, 1024, 2048]}   

    layers = block_layers[num_layers]
    filters = block_filters[num_layers]
    input = Input(input_shape)   

    x = Conv2D(64, kernel_size = 7, strides = 2, padding = "same")(input)
    
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(pool_size = (3,3), strides = (2,2))(x)
    x = resnet_first_module(x, filters[0])
    for a in range(4):
        num_layers = layers[a]
        num_filters = filters[a]      
        
        pool_first = True
        if a == 0:
            pool_first = False
            num_layers = num_layers - 1
        x = resnet_block(x, filters = num_filters, num_layers = num_layers, pool_first_layer = pool_first)   
        x = GlobalAveragePooling2D()(x) 
        x = Dense(num_classes)(x)
        x = Activation("softmax")(x)

        model = Model(inputs = input, outputs = x, name = "resnet_model{}".format(num_layers))

        return model
model = resnet_model(input_shape = (32,32,3), num_layers = 50, num_classes = 10)

model.summary()



(train_x, train_y), (test_x, test_y) = cifar10.load_data()

train_x = train_x.astype('float32')/255
test_x = test_x.astype('float32')/255

train_x = train_x - train_x.mean()
test_x = test_x - test_x.mean()

train_x = train_x/train_x.std()
test_x = test_x/test_x.std()

train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)        



def lr_train(epoch):
	lr = 0.001

	if epoch > 10:
		lr = lr/5
	elif epoch > 15:
		lr = lr/10
	elif epoch > 20:
		lr = lr/20

	return lr
lr_scheduler = LearningRateScheduler(lr_train)

save_directory = os.path.join(os.getcwd(), "cifar10savedmodels")

model_name = 'cifar10model.{epoch:03d}.h5'

if not os.path.isdir(save_directory):
	os.makedirs(save_directory)

model_path = (save_directory, model_name)

check_save = ModelCheckpoint(filepath = model_path, save_best_only = True, verbose = 1, monitor = "val_acc")

model.compile(optimizer = Adam(lr_train(0)), loss = "categorical_crossentropy", metrics = ["accuracy"])

batch_size = 32
epochs = 30
steps_per_epoch = ceil(50000/32)

datagen = ImageDataGenerator(rotation_range = 20, width_shift_range = 5/32, height_shift_range = 5/32)

model.fit_generator(datagen.flow(train_x, train_y, batch_size = batch_size), steps_per_epoch = steps_per_epoch, epochs = epochs, 
 callbacks = [lr_scheduler, check_save], validation_data = [test_x, test_y], verbose = 1)

accuracy = model.evaluate(test_x, test_y, batch_size = batch_size)
print("Accuracy: ", accuracy[1])

