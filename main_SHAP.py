import cv2
from PerceptSent import NeuralNetwork
from PerceptSent import Dataset
import shap
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt






DATASET = "PerceptSent/data"
HEATMAP = None
INPUT = "experiments_ensemble.csv"
OUTPUT = "output_ensemble_2"
PROFILING = False
FREEZE = False
EARLY_STOP = False
PERMUTATION = False
ENSEMBLE = True
K = 1
EPOCHS = 1

def crop_resize(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    y, x, c = img.shape
    if y < x:
        aux = int((x-y)/2)
        img = img[:,aux:aux+y]
    elif x < y:
        aux = int((y-x)/2)
        img = img[aux:aux+x,:]
    return np.float32(cv2.resize(img, (224, 224)))#/255.

if __name__ == '__main__':

    with open(INPUT, 'r') as e:
        lines = e.readlines()[1:]
        for line in lines:
            cfg = line.replace("\n", "").split(";")
            print(f"\n\n--- Executing experiment {str(cfg[0])} ---\n\n")
            try:

                dataset = Dataset(cfg, DATASET, OUTPUT, PROFILING)
                dataset.create()
                neural_network = NeuralNetwork(dataset, FREEZE, EARLY_STOP)
                X, Y = neural_network.load_dataset()

                y = [sum(y) for y in Y]
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 42)
                neural_network.train_model(X_train, Y_train, k=K, epochs=EPOCHS, permutation=PERMUTATION)
                
                Xtr, Ytr = neural_network.unpack(X_train, Y_train)                
                X_img = list()
                X_att = list()
                for img in Xtr:
                    img_data = crop_resize(img)
                    X_img.append(img_data)
                    if dataset.attributes is not None:
                        img_name = ((img.split('/')[-1]).split('.')[0])
                        a = f"{dataset.attributes}/{img_name}.txt"
                        att = np.loadtxt(a)
                        X_att.append(att)
                if dataset.attributes is not None:
                    X_train_b = [np.asarray(X_img), np.asarray(X_att)]
                else:
                    X_train_b = np.asarray(X_img)

                Xte, Yte = neural_network.unpack(X_test, Y_test)  
                for img in Xte:
                    img_data = crop_resize(img)
                    X_img.append(img_data)
                    if dataset.attributes is not None:
                        img_name = ((img.split('/')[-1]).split('.')[0])
                        a = f"{dataset.attributes}/{img_name}.txt"
                        att = np.loadtxt(a)
                        X_att.append(att)
                if dataset.attributes is not None:
                    X_test_b = [np.asarray(X_img), np.asarray(X_att)]
                else:
                    X_test_b = np.asarray(X_img)
                
                b = X_train_b.shape[0]
                background = X_train_b[np.random.choice(b, 1000, replace=False)]
                explainer = shap.DeepExplainer(neural_network.model, background)
                  
                # example image for each class for test set
                x_test_dict = dict()
                for i, l in enumerate(Yte):
                    if len(x_test_dict)==3:
                        break
                    if l not in x_test_dict.keys():
                        x_test_dict[l] = X_test_b[i]

                # order by class
                x_test_each_class = [x_test_dict[i] for i in sorted(x_test_dict)]
                x_test_each_class = np.asarray(x_test_each_class)
                y_test_each_class = [0, 1, 2]

                # Compute predictions
                predictions = neural_network.model.predict(x_test_each_class)
                predicted_class = np.argmax(predictions, axis=1)

                # compute shap values
                shap_values = explainer.shap_values(x_test_each_class)

                pred_labels = [dataset.classes_list[i] for i in predicted_class]
                true_labels = [dataset.classes_list[l] for l in y_test_each_class]

                shap.image_plot(shap_values=shap_values, pixel_values=x_test_each_class * 255, labels=pred_labels, true_labels=true_labels, show=False)
                plt.savefig('SHAP_perceptsent.png')

            except Exception as e:
                print(f"\n\n\nSorry, something went wrong in experiment {str(cfg[0])}: {e}\n\n\n")
                with open(f"{OUTPUT}/log.txt", 'a+') as f:
                    f.write(str(e))
            else:
                print(f"\n\nExperiment {str(cfg[0])} successfully completed!\n\n") 