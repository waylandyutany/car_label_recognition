from tensorflow.keras import datasets, layers, models
import glob2, os, cv2, json
import numpy as np

################################################################################
class LetterRecognitor:
    @staticmethod
    def load_or_create(folder, rc_dimms):
        (rc_width, rc_height) = rc_dimms

        file_path = os.path.join(folder, "{}_{}.h5".format(rc_width, rc_height))
        ret = LetterRecognitor(rc_width, rc_height)
        if os.path.isfile(file_path):
            ret.load(file_path)
        else:
            learn_folder_name = os.path.join(folder, "{}_{}".format(rc_width, rc_height))
            ret.learn_from_samples(learn_folder_name)
            ret.save(file_path)
        return ret

################################################################################
    def __init__(self, rc_width, rc_height):
        self.__rc_width = rc_width
        self.__rc_height = rc_height
        self.__model = None

        self.__labels = []

    #recognition dimmensions (width, height)
    @property
    def rc_dimms(self):
        return (self.__rc_width, self.__rc_height)

    def learn_from_samples(self, folder_name):
        unique_labels = []
        label_indexes = []
        images = []
        file_names = glob2.glob(os.path.join(folder_name,"*.jpg"))
        for file_name in file_names:
            name,_ = os.path.splitext(os.path.basename(file_name))
            label = name.split('_')[0]
            images.append(cv2.imread(file_name,cv2.IMREAD_GRAYSCALE))
            if label not in unique_labels:
                unique_labels.append(label)
                self.__labels.append(label)
            label_indexes.append(unique_labels.index(label))

        train_images = np.array(images)
        train_images = train_images.reshape((len(label_indexes), self.__rc_width, self.__rc_height, 1))
        train_images = train_images + 100.0
        train_images = train_images / 355.0

        labels_count = len(unique_labels)
        train_labels = np.array(label_indexes)

        self.__model = models.Sequential()
        self.__model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.__rc_width, self.__rc_height, 1)))
        self.__model.add(layers.MaxPooling2D((2, 2)))
        self.__model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.__model.add(layers.MaxPooling2D((2, 2)))
        self.__model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.__model.add(layers.Flatten())
        self.__model.add(layers.Dense(32, activation='relu'))
        self.__model.add(layers.Dense(labels_count, activation='softmax'))
        self.__model.summary()
        self.__model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

        self.__model.fit(train_images, train_labels, epochs=30)

        test_images, test_labels = train_images, train_labels
        test_loss, test_acc = self.__model.evaluate(test_images, test_labels)
        print(test_acc)
        result = np.argmax(self.__model.predict(test_images), axis=1).tolist()
        print(result)

    def save(self, file_name):
        json_dict = {'labels' : self.__labels}

        with open(file_name+".json", 'w') as f:
            json.dump(json_dict, f)
        self.__model.save(file_name)

    def load(self, file_name):
        self.__model = models.load_model(file_name)
        with open(file_name+".json") as f:
            json_dict = json.load(f)

        self.__labels = json_dict['labels'] 

    def detect_images(self,images):
        if len(images) == 0:
            return []

        predict_images = np.array(images)
        predict_images = predict_images.reshape((len(images), self.__rc_width, self.__rc_height, 1))
        predict_images = predict_images + 100.0
        predict_images = predict_images / 355.0
        result = np.argmax(self.__model.predict(predict_images), axis=1).tolist()
        return [self.__labels[result[i]] for i in range(0,len(result))]


