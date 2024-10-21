import matplotlib.pyplot as plt
from logistic_regression_utils import *
from pre_process import *
from PIL import Image


def plot_learning_curve(model):
    """
    Plot the learning curve of a model
    :param model: the input model
    :return: the learning curve of the input model
    """
    cost = np.squeeze(model["costs"])
    plt.plot(cost)
    plt.ylabel("cost")
    plt.xlabel("iterations (per hundreds)")
    plt.title("Learning rate = " + str(model["learning_rate"]))
    plt.show()


def model_training():
    """
    A method to train model
    :return: the logistic regression trained model
    """
    train_set_x, train_set_y, test_set_x, test_set_y, classes = load_split_data()
    logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, 2000, 0.005, True)
    return logistic_regression_model


def test_image(image_fn, logistic_model):
    num_pixel = 64
    image = np.array(Image.open(image_fn).resize((64, 64)))
    plt.imshow(image)
    image = image / 255
    image = image.reshape((1, num_pixel * num_pixel * 3)).T
    predicted_image = predict(logistic_model["w"], logistic_model["b"], image)
    if predicted_image == 1:
        print("y = " + str(np.squeeze(predicted_image)) + ", your algorithm predicts a \"" + "Cat" "\" picture.")
    else:
        print("y = " + str(np.squeeze(predicted_image)) + ", your algorithm predicts a \"" + "Non-Cat" "\" picture.")


model_train = model_training()
plot_learning_curve(model_train)


