
from PerceptSent.Dataset import Dataset

DATASET = "PerceptSent/data/dataset.json"
INPUT = "experiments.csv"
OUTPUT = "output"
PROFILING = False

if __name__ == '__main__':

    with open(INPUT, 'r') as e:
        lines = e.readlines()[1:]
        for line in lines:
            cfg = line.replace("\n", "").split(";")
            dataset = Dataset(cfg, DATASET, OUTPUT, PROFILING)
            dataset.create()


# import os
# import time
# import json
# from OutdoorSent import OutdoorSent

# K = 5
# EPOCHS = 20

# def run_experiment(arq, imgs, folder, folder_data_augmented, att, early_stop, freeze, data_augmented, n_atts, n_classes, k, epochs, classes, output_folder):
#     outdoorsent = OutdoorSent(model=arq, attributes=att, n_atts=n_atts, n_classes=n_classes,
#                               freeze = freeze, early_stop=early_stop, classes=classes, 
#                               data_augmented=data_augmented, output_folder=output_folder)
#     X, Y = outdoorsent.load_dataset(imgs, folder, folder_data_augmented, False)
#     outdoorsent.unpack(X, Y, data_augmented, "imgs")
#     outdoorsent.train_model(X, Y, k=k, epochs=epochs)
#     del outdoorsent, X, Y

# if __name__ == '__main__':

#     image_folder = "images2"
#     experiment = "experiments.txt"

#     with open(f"{image_folder}/setup/{experiment}", 'r') as e:
#         lines = e.readlines()
#         for line in lines:
#             exp = line.replace("\n", "").split(";")
#             id = int(exp[0])
#             with open(f"{image_folder}/setup/{id}/metadata.txt", 'r') as e:
#                 cfg = json.loads(e.readline())
#                 classes = e.readline().replace("\n", "").split(",")
#             output_folder = create_dir(f"output/{str(id)}")

#             # check for descriptors.
#             if cfg.get("descriptors"):
#                 att = f"{image_folder}/setup/{id}"
#                 if cfg.get("method") == "multi":
#                     informations = {"age": 1, "gs": 1, "eco": 1, "edu": 1, "opt1": 1, "neg1": 1, "opt2": 1, "reasons": 37}
#                 elif cfg.get("method") == "multi_tag":
#                     informations = {"age": 1, "gs": 1, "eco": 1, "edu": 1, "opt1": 1, "neg1": 1, "opt2": 1, "reasons": 146}
#                 else:
#                     informations = {"age": 5, "gs": 3, "eco": 4, "edu": 4, "opt1": 6, "neg1": 6, "opt2": 6, "reasons": 37}
#                 n_atts = 0
#                 for desc_label in cfg.get("descriptors"):
#                     n_atts += informations.get(desc_label)
#             else:
#                 att = None
#                 n_atts = 0

#             # Check the folder to retrieve the images.
#             if cfg.get("method") == "multi":
#                 folder = "images2/multi"
#             else:
#                 folder = "images2/original_multi"
            
#             # Check if data is augmented.
#             if cfg.get("data_augmented"):
#                 data_augmented = True   
#                 folder_data_augmented = "images2/original_multi"
#             else:
#                 data_augmented = False   
#                 folder_data_augmented = None          

#             print(f"--- Executando experimento {str(id)} ---\n")
#             start_time = time.time()
#             try:
#                 run_experiment(arq="none", imgs=f"{image_folder}/setup/{id}/data.txt", folder=folder, folder_data_augmented=folder_data_augmented, 
#                                 att=att, early_stop=False, freeze=False, data_augmented = data_augmented, n_atts=n_atts, n_classes=len(classes), 
#                                 k=K, epochs=EPOCHS, 
#                                 classes=classes, output_folder=output_folder)
#             except Exception as e:
#                 print(f"Algo deu errado: {e}\n")
#                 with open(f"{output_folder}/log.txt", 'a+') as f:
#                     f.write(str(e))
#             else:
#                 print(f"Executado com sucesso!\n")
#             print(f"--- Concluído em {time.time() - start_time} segundos ---\n")    

# from make_dataset import generate_database, create_dir, config