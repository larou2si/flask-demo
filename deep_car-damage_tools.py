def preprocess_input_vgg(x):
    """Wrapper around keras.applications.vgg16.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.
    
    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)
    
    Note we cannot pass keras.applications.vgg16.preprocess_input()
    directly to to keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.
    
    Returns a numpy 3darray (the preprocessed image).
    
    """
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]
	
def save_bottleneck_features(location):    # SAMAJANA
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg)
    
    if(os.path.exists(location+'/bottleneck_features_train.npy')):
        print('Already exists',location+'/bottleneck_features_train.npy')
    else:
        train_generator = datagen.flow_from_directory(train_data_dir,
                                                      target_size=(img_height,img_width),
                                                      batch_size=16,
                                                      class_mode=None,         # only data, no labels
                                                      shuffle=False)           # keep data in same order as labels
    
        bottleneck_features_train = base_model.predict_generator(train_generator,
                                                                 nb_train_samples / 16,
                                                                 verbose=1)
        
        print('Saving',location+'/bottleneck_features_train.npy')
        np.save(open(location+'/bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
        #np.save(location+'/bottleneck_features_train.npy',bottleneck_features_train)
    
    if(os.path.exists(location+'/bottleneck_features_val.npy')):
        print('Already exists',location+'/bottleneck_features_val.npy')
    else:
        # Repeat it with validation data
        val_generator = datagen.flow_from_directory(val_data_dir,
                                                    target_size=(img_height,img_width),
                                                    batch_size=16,
                                                    class_mode=None,
                                                    shuffle=False)

        bottleneck_features_val = base_model.predict_generator(val_generator,
                                                               nb_val_samples / 16,
                                                               verbose=1)
        print('Saving',location+'/bottleneck_features_val.npy')
        np.save(open(location+'/bottleneck_features_val.npy','wb'),bottleneck_features_val)
        #np.save(location+'/bottleneck_features_val.npy',bottleneck_features_train)



# load saved data and train a small, fully-connected ModelCheckpoint
def train_categorical_model(location):
    # the features were saved in order, so recreating the labels is not hard
    train_data = np.load(open(location+'/bottleneck_features_train.npy', 'rb'))
    print(train_data.shape[1:])
    train_labels = np.array([0]*train_samples[0]
                            +[1]*train_samples[1]
                            +[2]*train_samples[2])
    
    train_labels = to_categorical(train_labels)
    
    val_data = np.load(open(location+'/bottleneck_features_val.npy','rb'))
    val_labels = np.array([0]*val_samples[0]
                          +[1]*val_samples[1]
                          +[2]*val_samples[2])
    
    val_labels = to_categorical(val_labels)
    
    model = Sequential()
    model.add(Flatten(input_shape=(train_data.shape[1:])))     # 8, 8, 512
    model.add(Dense(units=256,activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=3,activation='softmax'))             # upped to 3 so activation softmax
    
    model.compile(optimizer=optimizers.SGD(lr=0.0001,momentum=0.9),    
                  loss='categorical_crossentropy',
                 metrics = ['accuracy'])
    
    checkpoint = ModelCheckpoint(top_model_weights_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,mode='auto')     # ?read documentation
    
    fit = model.fit(train_data,train_labels,
                    #epochs=nb_epoch,
                    epochs=60,
                    batch_size=16,
                    validation_data=(val_data,val_labels),
                    callbacks=[checkpoint])
    try:
        with open(location+'/top_history.txt', 'w') as f:
            json.dump(fit.history, f)
    except:
        pass
    
    return model, fit.history


def finetune_categorical_model(location):
    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(units=256,activation='relu'))
    top_model.add(Dropout(rate=0.5))
    top_model.add(Dense(units=3,activation='softmax'))
    
    top_model.load_weights(top_model_weights_path) # load weights_path
    
    #base_model.add(top_model)
    
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    #model.add(top_model)
    
#     # set the first 25 layers (up to the last conv block)
#     # to non-trainable - weights will not be updated
#     for layer in model.layers[:25]:
#         layer.trainable=False

    # compile the model with a SGD/momentum optimizer 
    # and a very slow learning rate
    model.compile(optimizer=optimizers.SGD(lr=0.00001,momentum=0.9),    # reduced learning rate by 1/10
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    
    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg)
    
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_height, img_width),
                                                        batch_size=8,
                                                        class_mode='categorical')
    
    test_generator = test_datagen.flow_from_directory(val_data_dir,
                                                      target_size=(img_height, img_width),
                                                      batch_size=8,
                                                      class_mode='categorical',
                                                      shuffle=False)
    checkpoint = ModelCheckpoint(filepath=fine_tuned_model_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto')
    
    # fine-tune the model
    fit = model.fit_generator(train_generator,
                              steps_per_epoch=nb_train_samples/8,
                              #epochs=nb_epoch,
                              epochs=10,
                              validation_data=test_generator,
                              validation_steps=nb_val_samples/8,
                              verbose=1,
                              callbacks=[checkpoint])

    with open(location+'/ft_history.txt', 'w') as f:
        json.dump(fit.history, f)
    
    return model, fit.history
    

def evaluate_categorical_model(model,directory,labels,Force=False):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg)
    
    if(os.path.exists(location+'/pred_labels.npy')) and not Force:
        print('Already exists',location+'/pred_labels.npy')
        pred_labels = np.load(open(location+'/pred_labels.npy', 'rb'))
    else:
        # Repeat it with validation data
        test_generator = test_datagen.flow_from_directory(directory,
                                                     target_size=(img_height,img_width),
                                                     batch_size=8,
                                                     class_mode='categorical',     # categorical for multiclass
                                                     shuffle=False)
        
        predictions = model.predict_generator(test_generator,
                                              steps=len(labels)/8,
                                              verbose=1)
        
        # use for multiclass
        pred_labels = np.argmax(predictions, axis=1)
    
        # pred_labels = [0 if i <0.5 else 1 for i in predictions]
        print('Saving',location+'/pred_labels.npy')
        np.save(open(location+'/pred_labels.npy','wb'),pred_labels)
        #np.save(location+'/bottleneck_features_val.npy',bottleneck_features_train)
    
    print()
    print(classification_report(labels, pred_labels))
    print()
    cm = confusion_matrix(labels, pred_labels)
#     sns.heatmap(cm, annot=True, fmt='g');
    return cm
    
def plot_metrics(hist, stop=50):  # stop -> no of data pts in plot
                                # hist(history) -> dict
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    # axes = axes.flatten()   # flatten -> numpy flatten

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    # axes = axes.flatten()   # flatten -> numpy flatten

    ax0.plot(range(60), d3a_history1.history['accuracy'], label='Training', color='#FF533D')
    ax0.plot(range(60), d3a_history1.history['val_accuracy'], label='Validation', color='#03507E')
    ax0.set_title('Accuracy')
    ax0.set_ylabel('Accuracy')
    ax0.set_xlabel('Epoch')
    ax0.legend(loc='lower right')

    ax1.plot(range(60), d3a_history1.history['loss'], label='Training', color='#FF533D')
    ax1.plot(range(60), d3a_history1.history['val_loss'],label='Validation', color='#03507E')
    ax1.set_title('Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper right')

    plt.tight_layout()

    print("Best Model: ")
    print_best_model_results(hist)
    
def view_images(img_dir,img_list):
    for img in img_list:
        clear_output()
        display(Image(img_dir+img))
        num = input("c to continue, q to quit")
        if num == 'c':
            pass
        else:
            return 'Finished for now.'

def print_best_model_results(model_hist):
    best_epoch = np.argmax(model_hist['val_acc'])
    print('epoch:', best_epoch+1,', val_acc:', model_hist['val_acc'][best_epoch],', val_loss:', 
          model_hist['val_loss'][best_epoch])













        
        
        
       