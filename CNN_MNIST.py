import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
from tensorflow.examples.tutorials.mnist import input_data
savedir = "D:\\OneDrive\\Study\\MachineLearning+ComputerVision using Python with Tensorflow\\TFT\\MNIST\\tmp\\"

# Data Load
data = input_data.read_data_sets("MNIST_data/", one_hot=True)

# One-Hot Coding --> Class
data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

# DB에 관한 Parameters
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

# Batch Size
train_batch_size = 100
batch_size = 256

# Validation Parameter
best_validation_accuracy = 0.0
last_improvement = 0
require_improvement = 2000
total_iterations = 0

def plot_images(plotid, images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        img_reshape = images[i].reshape(img_shape)
        ax.imshow(img_reshape, cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    # plt.show()
    fig.savefig(savedir+str(plotid)+".png")
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features
def optimize(num_iterations):
    # Ensure we update the global variables rather than local copies.
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):

        # Increase the total number of iterations performed.
        # It is easier to update it in each iteration because
        # we need this number several times in the following.
        total_iterations += 1

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch, y_true: y_true_batch, dropout_rate_cnn:0.7, dropout_rate_fc:0.5}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations and after last iteration.
        if (total_iterations % 100 == 0) or (i == (num_iterations - 1)):

            # Calculate the accuracy on the training-batch.
            feed_dict_acc = {x: x_batch, y_true: y_true_batch, dropout_rate_cnn: 1.0, dropout_rate_fc: 1.0}
            acc_train = session.run(accuracy, feed_dict=feed_dict_acc)

            # Calculate the accuracy on the validation-set.
            # The function returns 2 values but we only need the first.
            acc_validation, _ = validation_accuracy()

            # If validation accuracy is an improvement over best-known.
            if acc_validation > best_validation_accuracy:
                # Update the best-known validation accuracy.
                best_validation_accuracy = acc_validation

                # Set the iteration for the last improvement to current.
                last_improvement = total_iterations

                # Save all variables of the TensorFlow graph to file.
                saver.save(sess=session, save_path=save_path)

                # A string to be printed below, shows improvement found.
                improved_str = '*'
            else:
                # An empty string to be printed below.
                # Shows that no improvement was found.
                improved_str = ''

            # Status-message for printing.
            msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"

            # Print it.
            print(msg.format(i + 1, acc_train, acc_validation, improved_str))

        # If no improvement found in the required number of iterations.
        if total_iterations - last_improvement > require_improvement:
            print("No improvement found in a while, stopping optimization.")

            # Break out from the for-loop.
            break

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
def plot_example_errors(plotid, cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(plotid, images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
def plot_confusion_matrix(plotid, cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    #plt.show()
    plt.savefig(savedir + "ConfusionMatrix" + str(plotid) + ".png")
def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :], y_true: labels[i:j, :], dropout_rate_cnn:1.0, dropout_rate_fc:1.0}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred
def predict_cls_test():
    return predict_cls(images = data.test.images, labels = data.test.labels, cls_true = data.test.cls)
def predict_cls_validation():
    return predict_cls(images = data.validation.images,
                       labels = data.validation.labels,
                       cls_true = data.validation.cls)
def cls_accuracy(correct):
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / len(correct)

    return acc, correct_sum
def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    # The function returns two values but we only need the first.
    correct, _ = predict_cls_validation()

    # Calculate the classification accuracy and return it.
    return cls_accuracy(correct)
def print_test_accuracy(plotid, show_example_errors=False, show_confusion_matrix=False):
    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = cls_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(plotid, cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(plotid, cls_pred=cls_pred)
def plot_conv_weights(plotid, weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print mean and standard deviation.
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    # plt.show()
    fig.savefig(savedir + "Weights" +str(plotid) + ".png")
def plot_conv_layer(plotid, layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image], dropout_rate_cnn:1.0}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    # plt.show()
    fig.savefig(savedir + "Layers" + str(plotid) + ".png")

# Placeholders: X, Dropout Rate, Y
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
dropout_rate_cnn = tf.placeholder(tf.float32)
dropout_rate_fc = tf.placeholder(tf.float32)

# Networks Parameters
filtersize = 3
depth1 = 32
depth2 = 64
depth3 = 128
fc1size = 625
fc2size = 10

# weights_conv1 = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

# CNN Layer 1 [3 x 3 x numchannel x 32]
shape = [filtersize,filtersize,num_channels,depth1]
weights_conv1 = tf.get_variable("weights_conv1", shape=shape, initializer=tf.contrib.layers.xavier_initializer())

layer = tf.nn.conv2d(input=x_image, filter=weights_conv1, strides=[1, 1, 1, 1], padding='SAME')
layer = tf.nn.relu(layer)
layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
layer_conv1 = tf.nn.dropout(layer,dropout_rate_cnn)

# CNN Layer 2 [3 x 3 x 32 x 64]
shape = [filtersize,filtersize,depth1,depth2]
weights_conv2 = tf.get_variable("weights_conv2", shape=shape, initializer=tf.contrib.layers.xavier_initializer())

layer = tf.nn.conv2d(input=layer_conv1, filter=weights_conv2, strides=[1, 1, 1, 1], padding='SAME')
layer = tf.nn.relu(layer)
layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
layer_conv2 = tf.nn.dropout(layer,dropout_rate_cnn)

# CNN Layer 3 [3 x 3 x 64 x 128]
shape = [filtersize,filtersize,depth2,depth3]
weights_conv3 = tf.get_variable("weights_conv3", shape=shape, initializer=tf.contrib.layers.xavier_initializer())

layer = tf.nn.conv2d(input=layer_conv2, filter=weights_conv3, strides=[1, 1, 1, 1], padding='SAME')
layer = tf.nn.relu(layer)
layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
layer_conv3 = tf.nn.dropout(layer,dropout_rate_cnn)

# Flatten Process
layer_flat, num_features = flatten_layer(layer_conv3)

# FC Layer 1 [num_features - 128]
shape = [num_features, fc1size]
weights_fc1 = tf.get_variable("weights_fc1", shape=shape, initializer=tf.contrib.layers.xavier_initializer())

layer = tf.matmul(layer_flat, weights_fc1)
layer = tf.nn.relu(layer)
layer_fc1 = tf.nn.dropout(layer, dropout_rate_fc)

# FC Layer 2 [128 - 10]
shape = [fc1size, fc2size]
weights_fc2 = tf.get_variable("weights_fc2", shape=shape, initializer=tf.contrib.layers.xavier_initializer())

layer_fc2 = tf.matmul(layer_fc1, weights_fc2)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Softmax Layer for Accuracy Calculation
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

# Accuracy 식
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Best Validation 저장하는 Saver의 PATH 설정
saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')

# 세션 시작 및 초기화
session = tf.Session()
session.run(tf.global_variables_initializer())





# Visualization 샘플
image1 = data.test.images[0]

# WITHOUT Optimization
print_test_accuracy(1)
plot_conv_weights(1, weights=weights_conv1)
plot_conv_weights(2, weights=weights_conv2)
plot_conv_layer(1, layer=layer_conv1, image=image1)
plot_conv_layer(2, layer=layer_conv2, image=image1)

# Optimization, it store Best Validation
optimize(num_iterations=2000)

# 1차 Optimization 후
print_test_accuracy(1, show_example_errors=True, show_confusion_matrix=True)
plot_conv_weights(3, weights=weights_conv1)
plot_conv_weights(4, weights=weights_conv2)
plot_conv_layer(3, layer=layer_conv1, image=image1)
plot_conv_layer(4, layer=layer_conv2, image=image1)

# Reset (다시 없던 걸로)
session.run(tf.global_variables_initializer())
print_test_accuracy(2)
plot_conv_weights(5, weights=weights_conv1)
plot_conv_weights(6, weights=weights_conv2)
plot_conv_layer(5, layer=layer_conv1, image=image1)
plot_conv_layer(6, layer=layer_conv2, image=image1)

# Restore Best Validation Model
saver.restore(sess=session, save_path=save_path)

# 2차 Optimization 수행
optimize(num_iterations=1000)
print_test_accuracy(2, show_example_errors=True, show_confusion_matrix=True)
plot_conv_weights(7, weights=weights_conv1)
plot_conv_weights(8, weights=weights_conv2)
plot_conv_layer(7, layer=layer_conv1, image=image1)
plot_conv_layer(8, layer=layer_conv2, image=image1)