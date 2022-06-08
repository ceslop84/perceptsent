import os
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.preprocessing import image
import matplotlib.cm as cm
import shutil
from PerceptSent import NeuralNetwork

class HeatMap():

    def __init__(self, neural_network):
        self.neural_network = neural_network

    def __create_dir(self, dir):

        if os.path.isdir(dir):
            shutil.rmtree(dir)
        try:
            os.mkdir(dir)
            return dir
        except Exception as e:
            print(f"Creation of directory {dir} failed: {e}")
            exit(1)

    def __decode_predictions(self, predictions):
        # max_value = np.amax(predictions)
        max_index = np.where(predictions==np.amax(predictions))
        i = max_index[0][0]
        return [i, self.neural_network.dataset.classes_list[i], predictions[i]]

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

    def make_heatmap(self, images, predictions, images_list_file, folder=None, n_images=-1):
        
        images_list = list()
        with open(images_list_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                images_list.append(line.replace("\n",""))

        if folder is None:
            folder = self.neural_network.results_folder
        self.__create_dir(f"{folder}/Heatmap")
        if n_images>len(images) or n_images<0:
            n_images = len(images)

        for i in range(n_images):

            if images[i][0][2] not in images_list:
                continue

            predicted = self.__decode_predictions(predictions[i])
            img_path = f"{images[i][0][0]}/{images[i][0][2]}.jpg"

            try:
                # Generate class activation heatmap
                heatmap = self.__make_gradcam_heatmap(img_path)
                # Save the heatmap fused with the input image.
                self.__save_heatmap(img_path, predicted, heatmap, f"{folder}/Heatmap")
            except Exception as e_hm:
                print(f"Error while generating heatmap from image {img_path}.")

    def consolidate(self, input, output):

        def find(img_name, folder):
            for _, _, files in os.walk(folder):
                for f in files:
                    if img_name in f:
                        if "Positive" in f:
                            return f.replace("_Positive.jpg", ""), "Positive"
                        elif "Negative" in f:
                            return f.replace("_Negative.jpg", ""), "Negative"
                        elif "Neutral" in f:
                            return f.replace("_Neutral.jpg", ""), "Neutral"
                        else:
                            raise Exception()
                return None, None

        exp_list = list()
        with open(input, 'r') as e:
                lines = e.readlines()[1:]
                for line in lines:
                    exp_list.append(line[0])

        out_folder = self.__create_dir(f"{output}/heatmap")

        with open("output_hm/extract.csv", "r") as f:
            lines = f.readlines()
            for line in lines:
                img_name = line.replace("\n", "")
                self.__create_dir(f"{out_folder}/{img_name}")
                for e in exp_list:
                    folder = f"output_hm/{e}/results/Heatmap"
                    file, sent = find(img_name, folder)
                    if file:
                        src = f"{folder}/{file}_{sent}.jpg"
                        dst = f"{out_folder}/{img_name}/{e}_{sent}.jpg"
                        shutil.copyfile(src, dst)
                    else:
                        src = None
                        dst = Path(f"{out_folder}/{img_name}/{e}_ND.jpg")
                        dst.touch(exist_ok=True)
                    del file, sent, folder, src, dst

    def __make_gradcam_heatmap(self, img_path):

        # Processing input data.
        pred_index=None
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
        array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(array, axis=0)

        model = self.neural_network.model
        last_conv_layer_name = self.neural_network.last_conv_layer.name

        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        tf.keras.backend.clear_session()
        grad_model = Model([model.inputs], 
                           [model.get_layer(last_conv_layer_name).output, model.output])

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()