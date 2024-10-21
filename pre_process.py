import numpy as np
import h5py


def load_split_data():
    train_dataset = h5py.File('dataset/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('dataset/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    train_set_x_orig = reshape_data(train_set_x_orig)
    test_set_x_orig = reshape_data(test_set_x_orig)
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def visualize_data():
    train_set_X_orig, train_set_Y_orig,test_set_X_orig,_,_ = load_split_data()
    image_dim = train_set_X_orig[0].shape[0]
    print(f"The total number of training examples is: {train_set_X_orig.shape[0]}")
    print(f"The total number of testing examples is: {test_set_X_orig.shape[0]}")
    print(f"Size of each images is: {image_dim} x {image_dim} pixels")


def reshape_data(data):
    """
    Reshape data so that images of size (num_pixel, num_pixel, 3) turns into a single vector (num_pixel * num_pixel * 3, 1)
    :param data: The data to be reshaped
    :return: the flatten data, a vector
    """

    data_flatten = data.reshape(data.shape[0], -1).T
    # Normalize the data
    data_flatten = data_flatten / 255
    return data_flatten




