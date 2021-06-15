image_directory = 'binary/'

SIZE = 224
label = []  # Place holders to define add labels. We will add 0 to all class images and 1 to background

###############################################################################################


def create_dataset(label, image_directory,path, size, label, dataset):
    class_images = os.listdir(image_directory + path)
    for i, image_name in enumerate(class_images):
        if (image_name.split('.')[1] == 'png'):
            image = cv2.imread(image_directory + '16_train_data/' + image_name)
            image = image/255.0
            image = Image.fromarray(image, 'RGB')
            image = image.resize((SIZE, SIZE))
            dataset.append(np.array(image))
            label.append(0)

def create_dataset_wrapper(image_directory,paths,size):
    dataset = []  # Many ways to handle data, you can use pandas. Here, we are using a list format.
    label = []  # Place holders to define add labels. We will add 0 to all class images and 1 to background
    SIZE = 224
    [create_dataset(label, image_directory, path, size, dataset) for path in paths]
    dataset = np.array(dataset)
    label = np.array(label)
    return(label, dataset)

train_paths = ['16_train_data/','8_train_data/','12_train_data/','1_train_data/']
label_train, dataset_train = create_dataset_wrapper(image_directory, paths, size)

validation_paths = ['16_validation_data/','8_validation_data/','12_validation_data/','1_validation_data/']
label_val, dataset_val = create_dataset_wrapper(image_directory, validation_paths, size)

test_paths = ['16_test_data/','8_test_data/','12_test_data/','1_test_data/']
label_test, dataset_test = create_dataset_wrapper(image_directory, test_paths, size)