import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.preprocessing import image
import matplotlib.cm as cm
import shutil

class HeatMap():

    def __init__(self, neural_network=None, images_list_file=None):
        self.neural_network = neural_network
        images_list = list()
        if images_list_file:
            with open(images_list_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    images_list.append(line.replace("\n",""))
        self.__heatmap_list = images_list

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

    def __save_heatmap(self, img_path, true_value, predicted_value, heatmap, folder):

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

        name_ext = Path(img_path).name

        # Heatmap on original image
        heatmap_img = jet_heatmap
        heatmap_img = image.array_to_img(heatmap_img)
        hm_save_path = f"{folder}/{name_ext.split('.')[0]}_hm_t-{true_value}_p-{predicted_value}.{name_ext.split('.')[1]}"
        heatmap_img.save(hm_save_path)

        # Original image
        org_img = img_array
        org_img = image.array_to_img(org_img)
        org_save_path = f"{folder}/{name_ext.split('.')[0]}_org_t-{true_value}_p-{predicted_value}.{name_ext.split('.')[1]}"
        org_img.save(org_save_path)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * 0.5 + img_array
        superimposed_img = image.array_to_img(superimposed_img)
        save_path = f"{folder}/{name_ext.split('.')[0]}_t-{true_value}_p-{predicted_value}.{name_ext.split('.')[1]}"
        superimposed_img.save(save_path)

    def __create_heatmap(self, img_path, true_value, predicted_value, folder):
        try:
            # Generate class activation heatmap
            heatmap = self.__make_gradcam_heatmap(img_path)
            # Save the heatmap fused with the input image.
            self.__save_heatmap(img_path, true_value, predicted_value, heatmap, f"{folder}/Heatmap")
        except Exception as e_hm:
            print(f"Error while generating heatmap from image {img_path}.")       

    def split_data(self, X, Y):
        X_train = list()
        Y_train = list()
        X_hm = list()
        Y_hm = list()
        for img in zip(X, Y):
            img_name = img[0][0][2]
            if img_name in self.__heatmap_list:
                X_hm.append(img[0])
                Y_hm.append(img[1])
            else:
                X_train.append(img[0])
                Y_train.append(img[1])  
        return X_train, Y_train, X_hm, Y_hm

    def make_heatmap(self, classification):
        
        folder = self.neural_network.results_folder
        self.__create_dir(f"{folder}/Heatmap")

        for img in classification:
            img_name = img[0].split('/')[-1][:33]
            if img_name not in self.__heatmap_list:
                continue
            self.__create_heatmap(img[0], img[3], img[4], folder)

    def consolidate(self, heatmap, input, output):

        def find(img_name, folder):
            img_list = list()
            for _, _, files in os.walk(folder):
                for f in files:
                    if img_name in f:
                        rater_sent = f.replace(f"{img_name}", "")
                        img_list.append([img_name, rater_sent])
            return img_list

        exp_list = list()
        with open(input, 'r') as e:
                lines = e.readlines()[1:]
                for line in lines:
                    exp_list.append([line.split(';')[0],line.split(';')[5]])

        out_folder = self.__create_dir(f"{output}/heatmap_extract")

        img_list = list()
        with open(heatmap, "r") as f:
            lines = f.readlines()
            for line in lines:
                img_list.append(line.replace("\n", ""))
        for img_name in img_list:
            dst_folder = self.__create_dir(f"{out_folder}/{img_name}")
            for e in exp_list:
                dst_folder = self.__create_dir(f"{out_folder}/{img_name}/{e[1]}")
                folder = f"{output}/{e[0]}/results/Heatmap"
                files = find(img_name, folder)
                if files:
                    for f in files:
                        src = f"{folder}/{f[0]}{f[1]}"
                        dst = f"{dst_folder}/{e[0]}{f[1]}"
                        shutil.copyfile(src, dst)
                else:
                        src = None
                        dst = Path(f"{dst_folder}/{e[0]}_ND.jpg")
                        dst.touch(exist_ok=True)

    def __make_gradcam_heatmap(self, img_path):

        # Processing input data.
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        if self.neural_network.dataset.attributes is not None:
            name_ext = Path(img_path).name
            att_path = self.neural_network.dataset.attributes +'/'+name_ext.split(".")[0]+'.txt'
            att = np.loadtxt(att_path)
            att_array = np.expand_dims(att, axis=0)

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

            if self.neural_network.dataset.attributes is not None:
                last_conv_layer_output, preds = grad_model([img_array, att_array])
            else:
                last_conv_layer_output, preds = grad_model(img_array)


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