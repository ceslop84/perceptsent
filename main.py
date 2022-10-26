from PerceptSent import NeuralNetwork
from PerceptSent import HeatMap
from PerceptSent import Dataset
from PerceptSent import Results


DATASET = "PerceptSent/data"
HEATMAP = None #"csv/heatmap_extract.csv"
INPUT = "csv/experiments.csv" #"csv/experiments_hm.csv" "csv/experiments_perm.csv"
OUTPUT = "output"
PROFILING = False
FREEZE = False
EARLY_STOP = False
PERMUTATION = False
K = 1
EPOCHS = 20

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

                if HEATMAP:
                    heatmap = HeatMap(neural_network, HEATMAP)
                    X_train, Y_train, X_hm, Y_hm = heatmap.split_data(X, Y)
                    neural_network.train_model(X_train, Y_train, k=K, epochs=EPOCHS, permutation=PERMUTATION)
                    classification = neural_network.classify(X_hm, Y_hm, "heatmap")
                    heatmap.make_heatmap(classification)
                else:
                    neural_network.train_model(X, Y, k=K, epochs=EPOCHS, permutation=PERMUTATION)
            except Exception as e:
                print(f"\n\n\nSorry, something went wrong in experiment {str(cfg[0])}: {e}\n\n\n")
                with open(f"{OUTPUT}/log.txt", 'a+') as f:
                    f.write(str(e))
            else:
                print(f"\n\nExperiment {str(cfg[0])} successfully completed!\n\n") 

    if HEATMAP:
        heatmap = HeatMap()
        heatmap.consolidate(HEATMAP, INPUT, OUTPUT)

        results = Results(K, EPOCHS, INPUT, OUTPUT)
        results.consolidate()