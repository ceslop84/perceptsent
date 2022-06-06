import os
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.preprocessing import image
import matplotlib.cm as cm
from PerceptSent import NeuralNetwork


class HeatMap():

    def __init__(self, neural_network):
        self.neural_network = neural_network

    def __crop_resize(self, img_file):
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        y, x, c = img.shape
        if y < x:
            aux = int((x-y)/2)
            img = img[:,aux:aux+y]
        elif x < y:
            aux = int((y-x)/2)
            img = img[aux:aux+x,:]
        return np.float32(cv2.resize(img, (self.neural_network.img_size, self.neural_network.img_size)))/255.

    def __create_dir(self, dir):
        if not os.path.isdir(dir):
                try:
                    os.mkdir(dir)
                    return dir
                except Exception as e:
                    print(f"Creation of directory {dir} failed: {e}")
                    exit(1)
        else:
            return dir

    def __decode_predictions(self, predictions):
        max_value = np.amax(predictions)
        max_index = np.where(predictions==np.amax(predictions))
        i = max_index[0][0]
        return [i, self.neural_network.dataset.classes_list[i], predictions[i]]

    def __create_heatmap_model(self):

        # First, we create a model that maps the input image to the activations
        # of the last conv layer
        last_conv_layer = self.neural_network.last_conv_layer
        self.__last_conv_layer_model = Model(self.neural_network.model.inputs, last_conv_layer.output)

        # Second, we create a model that maps the activations of the last conv
        # layer to the final class predictions
        input_tensor = Input(shape=last_conv_layer.output.shape[1:])
        x = input_tensor
        for layer_name in self.neural_network.classifier_layer_names:
            x = self.neural_network.model.get_layer(layer_name)(x)

        self.__classifier_model = self.neural_network.add_dense_layers(input_tensor, x)

    def __generate_heatmap(self, data_array):

        # Processing input data.
        img_array = np.expand_dims(data_array[0], axis=0)
        if self.neural_network.dataset.attributes is not None:
            att_array = np.expand_dims(data_array[1], axis=0)

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
            if self.neural_network.dataset.attributes is not None:
                last_conv_layer_output = self.__last_conv_layer_model([img_array, att_array])
            else:
                last_conv_layer_output = self.__last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            # Compute class predictions
            if self.neural_network.dataset.attributes is not None:
                preds = self.__classifier_model([last_conv_layer_output, att_array])
            else:
                preds = self.__classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        heatmap = np.mean(last_conv_layer_output, axis=-1)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap

    def __save_heatmap(self, img_path, predicted, heatmap, folder):

        class_name = predicted[1]

        # We load the original image
        img = image.load_img(img_path)
        img_array = image.img_to_array(img)

        # We rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # We use RGB values of the colormap and create an image with RGB colorized heatmap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
        jet_heatmap = image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * 0.4 + img_array
        superimposed_img = image.array_to_img(superimposed_img)

        # Save the superimposed image
        name_ext = Path(img_path).name
        save_path = f"{folder}/{name_ext.split('.')[0]}_{class_name}.{name_ext.split('.')[1]}"
        superimposed_img.save(save_path)

    def make_heatmap(self, images, predictions, folder=None, n_images=-1):
        if folder is None:
            folder = self.neural_network.results_folder
        self.__create_dir(f"{folder}/Heatmap")
        if n_images>len(images) or n_images<0:
            n_images = len(images)

        self.__create_heatmap_model()

        for i in range(n_images):
            predicted = self.__decode_predictions(predictions[i])

            img_path = f"{images[i][0][0]}/{images[i][0][2]}.jpg"
            img = self.__crop_resize(img_path)

            if self.neural_network.dataset.attributes is not None:
                name_ext = Path(img_path).name
                att_path = self.neural_network.attributes +'/'+name_ext.split(".")[0]+'.txt'
                att = np.loadtxt(att_path)
                X = [img, att]
            else:
                X = [img]

            try:
                # Generate class activation heatmap
                heatmap = self.__generate_heatmap(X)
                # Save the heatmap fused with the input image.
                self.__save_heatmap(img_path, predicted, heatmap, f"{folder}/Heatmap")

            except Exception as e_hm:
                print(f"Error while generating heatmap from image {img_path}.")
