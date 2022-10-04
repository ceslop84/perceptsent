from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from PerceptSent import NeuralNetwork
from PerceptSent import HeatMap
from PerceptSent import Dataset
from PerceptSent import Results


DATASET = "PerceptSent/data"
HEATMAP = "heatmap_extract.csv"
INPUT = "experiments_hm_2.csv"
OUTPUT = "output_hm_3"
PROFILING = False
FREEZE = False
EARLY_STOP = False
PERMUTATION = False
ENSEMBLE = True
K = 1
EPOCHS = 20

if __name__ == '__main__':

    with open(INPUT, 'r') as e:
        lines = e.readlines()[1:]
        counter = 0
        for index, line in enumerate(lines):
            if index != counter:
                continue
            try:
                cfg1 = lines[index].replace("\n", "").split(";")
                cfg2 = lines[index+1].replace("\n", "").split(";")
                cfg3 = lines[index+2].replace("\n", "").split(";")

                exp_id = f"{cfg1[0]}-{cfg2[0]}-{cfg3[0]}"
                print(f"\n\n--- Executing experiment {exp_id} ---\n\n")

                dataset1 = Dataset(cfg1, DATASET, OUTPUT, PROFILING)
                dataset2 = Dataset(cfg2, DATASET, OUTPUT, PROFILING)
                dataset3 = Dataset(cfg3, DATASET, OUTPUT, PROFILING)

                dataset1.create()
                dataset2.create()
                dataset3.create()

                neural_network1 = NeuralNetwork(dataset1, FREEZE, EARLY_STOP)
                neural_network2 = NeuralNetwork(dataset2, FREEZE, EARLY_STOP)
                neural_network3 = NeuralNetwork(dataset3, FREEZE, EARLY_STOP)

                heatmap1 = HeatMap(neural_network1, HEATMAP)
                heatmap2 = HeatMap(neural_network2, HEATMAP)
                heatmap3 = HeatMap(neural_network3, HEATMAP)

                X, Y = neural_network1.load_dataset()
                X_train, Y_train, X_hm, Y_hm = heatmap1.split_data(X, Y)
                #neural_network1.train_model(X_train, Y_train, k=K, epochs=EPOCHS, permutation=PERMUTATION)
                neural_network1.model.load_weights(f"{OUTPUT}/{cfg1[0]}/results/Weights/weights_1_1.h5", by_name=True)
                neural_network2.model.load_weights(f"{OUTPUT}/{cfg1[0]}/results/Weights/weights_1_1.h5", by_name=True)
                neural_network3.model.load_weights(f"{OUTPUT}/{cfg1[0]}/results/Weights/weights_1_1.h5", by_name=True)

                neural_network2.train_model(X_train, Y_train, k=K, epochs=EPOCHS, permutation=PERMUTATION)
                neural_network3.train_model(X_train, Y_train, k=K, epochs=EPOCHS, permutation=PERMUTATION)
                
                #classification1 = neural_network1.classify(X_hm, Y_hm, "heatmap")
                classification2 = neural_network2.classify(X_hm, Y_hm, "heatmap")
                classification3 = neural_network3.classify(X_hm, Y_hm, "heatmap")

                #heatmap1.make_heatmap(classification1)
                heatmap2.make_heatmap(classification2)
                heatmap3.make_heatmap(classification3)

                counter += 3

            except Exception as e:
                print(f"\n\n\nSorry, something went wrong in experiment {exp_id}: {e}\n\n\n")
                with open(f"{OUTPUT}/log.txt", 'a+') as f:
                    f.write(str(e))
            else:
                print(f"\n\nExperiment {exp_id} successfully completed!\n\n") 

    if HEATMAP:
        heatmap = HeatMap()
        heatmap.consolidate(HEATMAP, INPUT, OUTPUT)

        results = Results(K, EPOCHS, INPUT, OUTPUT)
        results.consolidate()
