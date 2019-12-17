#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#Keras tools used
from keras import optimizers

from keras.models import Sequential

from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.optimizers import SGD, RMSprop, adam

from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# In[2]:


train_dir = "./Training/"
test_dir = "./Test/"
train_label = pd.read_csv('training-labels.csv')
test_label = pd.read_csv('SampleSubmission.csv')


# In[3]:


test_label.sample(3)


# In[4]:


train_label['Drscore']=train_label['Drscore'].apply(str)


# In[5]:


datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

# In[54]:


train_generator=datagen.flow_from_dataframe(
dataframe=train_label,
directory=train_dir,
x_col="Filename",
y_col="Drscore",
subset="training",
batch_size=100,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(32,32))


# In[55]:


valid_generator=datagen.flow_from_dataframe(
dataframe=train_label,
directory=train_dir,
x_col="Filename",
y_col="Drscore",
subset="validation",
batch_size=100,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(32,32))


# In[56]:


test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=test_label,
directory=test_dir,
x_col="Id",
y_col=None,
batch_size=27,
seed=42,
shuffle=False,
class_mode=None,
target_size=(32,32))


from keras.applications.vgg16 import VGG16 as PTModel
from keras.applications.inception_resnet_v2 import InceptionResNetV2 as PTModel
from keras.applications.inception_v3 import InceptionV3 as PTModel
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.models import Model
in_lay = Input((512,512,3))
base_pretrained_model = PTModel(input_shape =  (512,512,3), include_top = False, weights = 'imagenet')
base_pretrained_model.trainable = False
pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
pt_features = base_pretrained_model(in_lay)
from keras.layers import BatchNormalization
bn_features = BatchNormalization()(pt_features)

# here we do an attention mechanism to turn pixels in the GAP on an off

attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.5)(bn_features))
attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
attn_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
attn_layer = Conv2D(1, 
                    kernel_size = (1,1), 
                    padding = 'valid', 
                    activation = 'sigmoid')(attn_layer)
# fan it out to all of the channels
up_c2_w = np.ones((1, 1, 1, pt_depth))
up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
               activation = 'linear', use_bias = False, weights = [up_c2_w])
up_c2.trainable = False
attn_layer = up_c2(attn_layer)

mask_features = multiply([attn_layer, bn_features])
gap_features = GlobalAveragePooling2D()(mask_features)
gap_mask = GlobalAveragePooling2D()(attn_layer)
# to account for missing values from the attention model
gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
gap_dr = Dropout(0.25)(gap)
dr_steps = Dropout(0.25)(Dense(128, activation = 'relu')(gap_dr))
out_layer = Dense(t_y.shape[-1], activation = 'softmax')(dr_steps)
retina_model = Model(inputs = [in_lay], outputs = [out_layer])
from keras.metrics import top_k_categorical_accuracy
def top_2_accuracy(in_gt, in_pred):
    return top_k_categorical_accuracy(in_gt, in_pred, k=2)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                           metrics = ['categorical_accuracy', top_2_accuracy])
model.summary()



#batch_size to train
#batch_size = 100
# number of output classes
#nb_classes = 5
# number of epochs to train
#nb_epoch = 10

# number of convolutional filters to use
#nb_filters = 32
# size of pooling area for max pooling
#nb_pool = 2
# convolution kernel size
#nb_conv = 3

# In[57]:


#model = Sequential()

#model.add(Conv2D(nb_filters, nb_conv, nb_conv,
#                        border_mode='valid',
#                        input_shape=(32, 32, 3)))
#convout1 = Activation('relu')
#model.add(convout1)
#model.add(Conv2D(nb_filters, nb_conv, nb_conv))
#convout2 = Activation('relu')
#model.add(convout2)
#model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
#model.add(Dropout(0.5))

#model.add(Flatten())
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(nb_classes))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adadelta')

#model.add(Conv2D(32, (3, 3), padding='same',
#                 input_shape=(32,32,3)))
#model.add(Activation('relu'))
#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Conv2D(64, (3, 3), padding='same'))
#model.add(Activation('relu'))
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(5, activation='softmax'))
#model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])


# In[58]:


STEP_SIZE_TRAIN=100 #train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=100 #valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=37 #test_generator.n//test_generator.batch_size

print(STEP_SIZE_TRAIN,STEP_SIZE_VALID,STEP_SIZE_TEST)


# In[59]:


model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10,verbose=2
)


# In[60]:

model.save('my_model.h5')

model.evaluate_generator(generator=valid_generator,
steps=STEP_SIZE_TEST)


# In[61]:


test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=2)


# In[62]:


predicted_class_indices=np.argmax(pred,axis=1)


# In[63]:


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[66]:

len(predictions)


filenames=test_generator.filenames
results=pd.DataFrame({"Id":filenames,
                      "Expected":predictions})
results.to_csv("results.csv",index=False)







