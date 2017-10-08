import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np
import time


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    graph = tf.get_default_graph()
    img = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    #print("Tensor shape: ", l7.get_shape().as_list())
    
    return img, keep, l3, l4, l7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Matching the num_outputs to add skip coonection
    num_outputs = 512
    conv_1d = tf.layers.conv2d(vgg_layer7_out, num_outputs, 1, padding='same',
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output = tf.layers.conv2d_transpose(conv_1d, num_outputs, 4, 2, padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    # First skip connection from Layer 4
    output = tf.add(output, vgg_layer4_out)
    
    num_outputs = 256
    k_size = 4
    stride = (2, 2)
    output = tf.layers.conv2d_transpose(output, num_outputs, k_size, stride, padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    # Second skip connection from Layer 3
    output = tf.add(output, vgg_layer3_out)
    
    # Final output Layer with 2 output layers
    k_size = 16
    stride = (8, 8)
    output = tf.layers.conv2d_transpose(output, num_classes, k_size, stride, padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                                labels=labels))
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
    train_op = optimizer.minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement 
    init = tf.global_variables_initializer()
    sess.run(init)
    l_list = []
    print("Starting training...")
    for e in range(epochs):
        batch_count = 0
        for x, y in get_batches_fn(batch_size):
            batch_count += 1
            _, l = sess.run([train_op, cross_entropy_loss],
                            feed_dict = {input_image: x,
                                         correct_label: y,
                                         keep_prob: 0.75})
            # Crating list of losses for each batch
            l_list.append(l)

            # printing output to the terminal after every 3 batches
            if not batch_count%3:
                print("Batch {} loss: {:3.5f}".format(batch_count, l))
        # printing the output at the end of epoch
        print("\nLoss at end of epoch {}: {:3.5f}\n".format(e+1, l))
    
    # returning list of losses
    return l_list

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        img, keep, l3, l4, l7 = load_vgg(sess, vgg_path)
        
        last_layer = layers(l3, l4, l7, num_classes)
        
        labels = tf.placeholder(tf.float32, 
                                shape=[None, 
                                       image_shape[0], 
                                       image_shape[1], 
                                       num_classes])
        learning_rate = 1e-4

        logits, train_op, cross_entropy_loss = optimize(last_layer, labels, 
                                                        learning_rate, num_classes)
        # TODO: Train NN using the train_nn function
        epochs = 30
        batch_size = 5
        
        loss = train_nn(sess, epochs, batch_size, get_batches_fn, train_op, 
                        cross_entropy_loss, img, labels, keep, learning_rate)
        
        
        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep, img)
        
        # Saving loss list in a csv file
        loss_file = runs_dir + '/loss_' + str(time.time()) + '.csv'
        
        print("Saving loss.. ")
        np.savetxt(loss_file, loss, delimiter=',', fmt='%3.5f')
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
