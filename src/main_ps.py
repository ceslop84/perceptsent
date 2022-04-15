from make_dataset import generate_database, create_dir, config

if __name__ == '__main__':

    file_in = "final.json"
    folder_out = "output"
    exp_in = "experiments.txt"

    with open(exp_in, 'r') as e:
        lines = e.readlines()
        for line in lines:
            exp = line.replace("\n", "").split(";")
            id = int(exp[0])
            nr_classes = int(exp[1])
            expand_neutral = (exp[2]=="True")
            shift_neutral = str(exp[3])
            method = str(exp[4])
            data_augmented = (exp[5]=="True")
            metadata = (exp[6]=="True")
            semantic = (exp[7]=="True")
            filter = (exp[8]=="True")
            dataset_strategy = str(exp[9])
            if exp[10]: 
                dataset_size = int(exp[10])
            else:
                dataset_size = 5000
            class_balance = float(exp[11].replace(",","."))
            include_no_data = (exp[12]=="True")

            folder = create_dir(f"{folder_out}/{exp[0]}")
            cfg = config(nr_classes, expand_neutral, shift_neutral, method, 
                         data_augmented, metadata, semantic, filter, 
                         dataset_strategy, dataset_size,
                         class_balance, include_no_data)
            generate_database(file_in, folder, cfg)
