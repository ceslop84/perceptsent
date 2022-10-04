from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from PerceptSent import NeuralNetwork
from PerceptSent import HeatMap
from PerceptSent import Dataset
from PerceptSent import Results


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
                if ENSEMBLE:
                    y = [sum(y) for y in Y]
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 42)
                    neural_network.train_model(X_train, Y_train, k=K, epochs=EPOCHS, permutation=PERMUTATION)
                    if not isinstance(ENSEMBLE, list):
                        ENSEMBLE = list()
                    ENSEMBLE.append(neural_network.classify(X_test, Y_test))
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

    if ENSEMBLE:
        votes = len(ENSEMBLE)
        y_pred = list()
        y_true = list()
        pred = list()
        imgs = list()
        for i, e in enumerate(ENSEMBLE[0]):
            pool = [0, 0, 0]
            for j in range(votes):
                if ENSEMBLE[j][i][4] == 'Negative':
                    pool[0]+= 1
                elif ENSEMBLE[j][i][4] == 'Neutral':
                    pool[1]+= 1
                elif ENSEMBLE[j][i][4] == 'Positive':
                    pool[2]+= 1
                else:
                    raise Exception()
                

            if ENSEMBLE[j][i][3] == 'Negative':
                true = 0
            elif ENSEMBLE[j][i][3] == 'Neutral':
                true = 1
            elif ENSEMBLE[j][i][3] == 'Positive':
                true = 2
            else:
                raise Exception()
                
            y_pred.append(pool.index(max(pool)))
            y_true.append(true)
            pred.append(pool)
            imgs.append(ENSEMBLE[j][i][0])
                
        with open(f"{neural_network.results_folder}/test_ensemble.txt", 'a+') as f:
            f.write(str(classification_report(y_true, y_pred)))
            f.write("\n")
            f.write(str(accuracy_score(y_true, y_pred)))
            f.write("\n")
            f.write(str(confusion_matrix(y_true, y_pred)))
            f.write("\n")
            f.write("image, value, true_label, predict_label\n")
            for img, p, c, t in zip(imgs, pred, y_pred, y_true):
                f.write(f"{img}, {p}, {neural_network.dataset.classes_list[t]}, {neural_network.dataset.classes_list[c]}\n")
                    
