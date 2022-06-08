from PerceptSent import NeuralNetwork
from PerceptSent import HeatMap
from PerceptSent import Dataset
from PerceptSent import Results


DATASET = "PerceptSent/data"
HEATMAP = "heatmap.csv"
INPUT = "experiments_hm.csv"
OUTPUT = "output_hm"
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
                neural_network.train_model(X, Y, k=K, epochs=EPOCHS, permutation=PERMUTATION)

                if HEATMAP:
                    predict, _ = neural_network.classify(X, Y, "heatmap")
                    heatmap = HeatMap(neural_network)
                    heatmap.make_heatmap(X, predict, HEATMAP)

            except Exception as e:
                print(f"\n\n\nSorry, something went wrong in experiment {str(cfg[0])}: {e}\n\n\n")
                with open(f"{OUTPUT}/log.txt", 'a+') as f:
                    f.write(str(e))
            else:
                print(f"\n\nExperiment {str(cfg[0])} successfully completed!\n\n") 

    heatmap = HeatMap()
    heatmap.consolidate(INPUT, OUTPUT)

    results = Results(K, EPOCHS, INPUT, OUTPUT)
    results.consolidate()