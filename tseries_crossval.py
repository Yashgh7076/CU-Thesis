import numpy as np 
import sys
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF info
import tensorflow as tf
#import matplotlib.pyplot as plt

# Define constants
stride = 15 #1 second @ 15 Hz sampling
window = 30*15 #30 seconds window considered

folder = sys.argv[1] 
if not os.path.exists(folder):
    print("Unable to open folder containing data, check that folder exists \n")
    exit(0)
total_files = 488

total_sum = 0
for i in range(1,total_files + 1):
    file_no = 'output' + str(i) + '.txt'
    full_path = os.path.join(folder, file_no)
    #print(full_path)
    
    f = open(full_path,'r')
    d=[[float(x) for x in line.split()] for line in f]
    f.close()

    N = len(d) 
    total_sum = total_sum + N    

    M = len(d[0])
    measurements = int((M-window)/6) 
  
dataset = np.zeros((total_sum,measurements,6))
vectors = np.zeros((total_sum,window),dtype=np.uint8)
windows_in_recording = np.zeros((total_files), dtype=np.uint32)

total_windows = 0
for i in range(1,total_files + 1):
    file_no = 'output' + str(i) + '.txt'
    full_path = os.path.join(folder, file_no)
    
    f = open(full_path,'r')
    d=[[float(x) for x in line.split()] for line in f]
    f.close()    

    # Need to recalculate the number of windows each time
    N = len(d)
    
    labels = np.zeros(shape = (N,window), dtype=np.uint8) # np.uint8 -> each sample is labeled from 0 to 5
    data = np.zeros(shape = (N,measurements,6))
    data_max = np.zeros((6)) # Create placeholders
    data_min = np.zeros((6))
    temp_3 = np.zeros((6))
    temp_4 = np.zeros((6))

    for j in range(N):
        temp = d[j]
        temp_1 = temp[0:window]
        temp_2 = temp[window:M]   

        labels[j,:] = temp_1

        for k in range(measurements): # Read data
            for l in range(6):
                data[j,k,l] = temp_2[(6*k) + l] 
       
    for j in range(N):
        if(j == 1):
            data_max = np.amax(data[j,:,:], axis=0)
            data_min = np.amin(data[j,:,:], axis=0)
        else:
            temp_3 = np.amax(data[j,:,:], axis=0)
            temp_4 = np.amin(data[j,:,:], axis=0)
            for k in range(6):
                if(temp_3[k] >= data_max[k]):
                    data_max[k] = temp_3[k]

                if(temp_4[k] <= data_min[k]):
                    data_min[k] = temp_4[k]

    # Normalize each recording (meal)
    for j in range(N):
        for k in range(measurements):
            data[j,k,:] = data[j,k,:] - data_min # Vector subtraction
            data[j,k,:] = data[j,k,:]/(data_max - data_min) # Element-wise division

    dataset[total_windows:total_windows + N, :, :] = data
    vectors[total_windows:total_windows + N,:] = labels    
    total_windows = total_windows + N
    windows_in_recording[i-1] = total_windows #Calculates all windows till this meal -> That is what we want!

# Clear variables from memory
del data, labels, d, temp_1, temp_2, temp_3, temp_4 

# Print out to verify
#f = open('segments_data.txt','w') 
#for j in range(measurements):
#    for k in range(6):
#        f.write("%f " % (dataset[0,j,k]))
#    f.write("\n") # --> correct way of newline in Python!
#f.close()

#f = open('segments_labels.txt','w')
#for j in range(total_windows):
#    for k in range(window):
#        f.write("%u " % (vectors[j,k]))
#    f.write("\n")
#f.close()

# Cross-validation starts here, split data into five parts, use validation_split (keras) for simplicity
part_1 = windows_in_recording[math.floor((0.2*total_files)) -1]
part_2 = windows_in_recording[math.floor((0.4*total_files)) -1]
part_3 = windows_in_recording[math.floor((0.6*total_files)) -1]
part_4 = windows_in_recording[math.floor((0.8*total_files)) -1]
for iter in range(5):    
    
    if(iter == 0):
        tst_data = dataset[0:part_1,:,:]        
        trn_data = dataset[part_1:total_windows,:,:]

        tst_vcts = vectors[0:part_1,:]
        trn_vcts = vectors[part_1:total_windows,:]        
    elif(iter == 1):
        tst_data = dataset[part_1:part_2,:,:]
        temp_1 = dataset[0:part_1,:,:]
        temp_2 = dataset[part_2:total_windows,:,:]
        trn_data = np.concatenate((temp_1, temp_2), axis=0)

        tst_vcts = vectors[part_1:part_2,:]
        temp_3 = vectors[0:part_1,:]
        temp_4 = vectors[part_2:total_windows,:]
        trn_vcts = np.concatenate((temp_3, temp_4), axis=0)        
    elif(iter == 2):
        tst_data = dataset[part_2:part_3,:,:]
        temp_1 = dataset[0:part_2,:,:]
        temp_2 = dataset[part_3:total_windows,:,:]
        trn_data = np.concatenate((temp_1, temp_2), axis=0)
        
        tst_vcts = vectors[part_2:part_3,:]
        temp_3 = vectors[0:part_2,:]
        temp_4 = vectors[part_3:total_windows,:]
        trn_vcts = np.concatenate((temp_3, temp_4), axis=0)
    elif(iter == 3):
        tst_data = dataset[part_3:part_4,:,:]
        temp_1 = dataset[0:part_3,:,:]
        temp_2 = dataset[part_4:total_windows,:,:]
        trn_data = np.concatenate((temp_1, temp_2), axis=0)
        
        tst_vcts = vectors[part_3:part_4,:]
        temp_3 = vectors[0:part_3,:]
        temp_4 = vectors[part_4:total_windows,:]
        trn_vcts = np.concatenate((temp_3, temp_4), axis=0)
    elif(iter == 4):
        tst_data = dataset[part_4:total_windows,:,:]
        trn_data = dataset[0:part_4,:,:] 

        tst_vcts = vectors[part_4:total_windows,:]
        trn_vcts = vectors[0:part_4,:]     

    # Reshape labels -> needed for keras compatibility 
    trn_size = trn_data.shape[0] 
    trn_vcts = np.reshape(trn_vcts, newshape=(trn_size, 1, window)) # Each vector is of size 1 x training_window => 1 x N image of labels

    # Neural network training starts here
    print("Creating model", iter, "here")
    inputs = tf.keras.layers.Input(shape=(measurements, 6))
    reshape = tf.keras.layers.Reshape((1, measurements, 6))(inputs) # Data is a 1 x 450 'image' of 6 channels
    # Downstream --> Encoder
    conv_1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(1,15), strides=1, padding='same', activation='linear')(reshape)
    bn_1 = tf.keras.layers.BatchNormalization(axis=3)(conv_1)
    act_1 = tf.keras.layers.ReLU()(bn_1)
    pool_1 = tf.keras.layers.MaxPool2D(pool_size=(1,2))(act_1)

    conv_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1,7), strides=1, padding='same', activation='linear')(pool_1)
    bn_2 = tf.keras.layers.BatchNormalization(axis=3)(conv_2)
    act_2 = tf.keras.layers.ReLU()(bn_2)
    pool_2 = tf.keras.layers.MaxPool2D(pool_size=(1,2))(act_2)

    conv_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,5), strides=1, padding='same', activation='linear')(pool_2)
    bn_3 = tf.keras.layers.BatchNormalization(axis=3)(conv_3)
    act_3 = tf.keras.layers.ReLU()(bn_3)
    pool_3 = tf.keras.layers.MaxPool2D(pool_size=(1,2))(act_3)

    # Upstream --> Decoder
    up_conv1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(1,5),padding='same',strides=(1,2),activation='linear')(pool_3)
    bn_4 = tf.keras.layers.BatchNormalization(axis=3)(up_conv1)
    act_4 = tf.keras.layers.ReLU()(bn_4)
    concat = tf.keras.layers.Concatenate()
    cc_1 = concat([act_4, pool_2])

    up_conv2 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(1,7),padding='same',strides=(1,2),activation='linear')(cc_1)
    bn_5 = tf.keras.layers.BatchNormalization(axis=3)(up_conv2)
    act_5 = tf.keras.layers.ReLU()(bn_5)
    pad_1 = tf.keras.layers.ZeroPadding2D(padding=((0,0),(0,1)))(act_5)
    cc_2 = concat([pad_1, pool_1])

    # Final Layer
    pen_ult = tf.keras.layers.Conv2DTranspose(filters=6,kernel_size=(1,3),strides=(1,2),activation='softmax')(cc_2)
    outputs = tf.keras.layers.Cropping2D(cropping=((0,0),(0,1)))(pen_ult)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits='True'), metrics=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits='True')])
    if(iter == 0):
        model.summary()

    # Store training sequence to .txt file
    training_log = 'crossval_fold_' + str(iter) + '.txt'
    csv_logger = tf.keras.callbacks.CSVLogger(training_log, append = True, separator=' ')
    print("Training for fold", iter)
    metrics = model.fit(trn_data, trn_vcts, epochs=200, validation_split= 0.2, verbose=2, callbacks=[csv_logger])
    print("Saving model for fold", iter)
    model_ID = 'crossval_modelID_' + str(iter) + '.h5'
    tf.keras.models.save_model(model,model_ID)
    #del model -> Most likely not needed....

##print("Predict")
##op = model.predict(dataset[0:10,:,:])
##print(op.shape)
##temp = op[0,:,:,:]
##temp = np.reshape(temp,(window, 6))
##for i in range(window):
##	print(temp[i,:], np.argmax(temp[i,:]))

