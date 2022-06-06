import os
import json
import csv

class Results():

    def __init__(self, k, epochs, input, output):

        self.input = input
        self.output = output
        self.k = k
        self.epochs = epochs

        self.results_csv = open(f"{output}/results.csv", 'w+', newline='')
        self.results = csv.writer(self.results_csv, delimiter=';')

        self.predict_csv = open(f"{output}/predict.csv", 'w+', newline='')
        self.predict = csv.writer(self.predict_csv, delimiter=';')

        self.results.writerow([
                            "id", "file", "k", "epochs",
                            "nr_classes", "expand_neutral", "shift_neutral", "method", "data_augmented",
                            "descriptors", "filter", "class_balance", "include_no_data",
                            'Negative_id', 'SlightlyNegative_id', 'Neutral_id', 'SlightlyPositive_id', 'Positive_id',
                            "0_label", "1_label", "2_label", "3_label", "4_label",
                            "0_count", "1_count", "2_count", "3_count", "4_count",
                            "0_perc", "1_perc", "2_perc", "3_perc", "4_perc",
                            "0_precision", "0_recall", "0_fscore", "0_support",
                            "1_precision", "1_recall", "1_fscore", "1_support",
                            "2_precision", "2_recall", "2_fscore", "2_support",
                            "3_precision", "3_recall", "3_fscore", "3_support",
                            "4_precision", "4_recall", "4_fscore", "4_support",
                            "accuracy", "accuracy_support",
                            "macro_avg_precision", "macro_avg_recall", "macro_avg_fscore", "macro_avg_support",
                            "weighted_avg_precision", "weighted_avg_recall", "weighted_avg_fscore", "weighted_avg_support",
                            "img_agr", "img_dis", "avg_img_wkr_agr","avg_img_wkr_dis", "fleiss", "cronbach", "imgs", "imgs_0", "imgs_1", "imgs_2", "imgs_3", "imgs_4"
                            ])
        self.predict.writerow([
                            "id", "k_fold", "img", "true_label", "predict_label", "predict_result"
                            ])

    def consolidate(self):

        with open(self.input, 'r') as e:
            lines = e.readlines()[1:]
            for line in lines:
                exp = line.replace("\n", "").split(";")
                num_seq = str(exp[0])
                folder_res = f"{self.output}/{num_seq}/results"
                folder_stp = f"{self.output}/{num_seq}/dataset"
                self.__process_folder(num_seq, folder_res, folder_stp)
                
            self.results_csv.close()
            self.predict_csv.close()
                    
    def __process_folder(self, num_seq, folder_res, folder_stp):

        def process_file_result(num_seq, k, epochs, file_data, file_metadata, csv_writer):

            file_data.seek(0)
            lines = file_data.readlines()   

            if "precision    recall  f1-score   support" not in lines[0]:
                return None

            with open(file_metadata, 'r') as e:
                cfg = json.loads(e.readline())
                classes = e.readline().replace("\n", "").split(",")
                classes_dict = json.loads(e.readline())
                classes_count = json.loads(e.readline())
                classes_perc = json.loads(e.readline())   
                metrics =  json.loads(e.readline())

            p = ["" for i in range(5)]
            r = ["" for i in range(5)]
            f = ["" for i in range(5)]
            s = ["" for i in range(5)]
            
            c0 = lines[2].split()
            p[0] = float(c0[1])
            r[0] = float(c0[2])
            f[0] = float(c0[3])
            s[0] = float(c0[4])

            c1 = lines[3].split()
            p[1] = float(c1[1])
            r[1] = float(c1[2])
            f[1]= float(c1[3])
            s[1] = float(c1[4])
            
            l = 5
            
            if len(classes) in [3,5]:
                c2 = lines[4].split()
                p[2] = float(c2[1])
                r[2] = float(c2[2])
                f[2] = float(c2[3])
                s[2] = float(c2[4])
                l = 6
            
            if len(classes) == 5:
                c3= lines[5].split()
                p[3] = float(c3[1])
                r[3] = float(c3[2])
                f[3] = float(c3[3])
                s[3] = float(c3[4])
                c4= lines[6].split()
                p[4] = float(c4[1])
                r[4] = float(c4[2])
                f[4] = float(c4[3])
                s[4] = float(c4[4])
                l = 8

            acc = lines[l].split()
            mac = lines[l+1].split()
            wav = lines[l+2].split()

            af = float(acc[1])
            ap = float(acc[2])
            mp = float(mac[2])
            mr = float(mac[3])
            mf = float(mac[4])
            ms = float(mac[5])
            wp = float(wav[2])
            wr = float(wav[3])
            wf = float(wav[4])
            ws = float(wav[5])

            metrics_list = list()
            for key in metrics.keys():
                metrics_list.append(metrics[key]["0"])

            csv_writer.writerow([
                                num_seq, file_data.name, k, epochs,
                                cfg.get("nr_classes",""), cfg.get("expand_neutral",""), cfg.get("shift_neutral",""), cfg.get("method",""), cfg.get("data_augmented",""), 
                                cfg.get("descriptors",""), cfg.get("filter",""), cfg.get("class_balance",""), cfg.get("include_no_data",""),
                                classes_dict.get("Negative",""), classes_dict.get("SlightlyNegative",""), classes_dict.get("Neutral",""), classes_dict.get("SlightlyPositive",""), classes_dict.get("Positive",""),
                                str(classes[classes_dict.get("Negative","")]), str(classes[classes_dict.get("SlightlyNegative","")]), str(classes[classes_dict.get("Neutral","")]), str(classes[classes_dict.get("SlightlyPositive","")]), str(classes[classes_dict.get("Positive","")]),
                                classes_count.get("0",""), classes_count.get("1",""), classes_count.get("2",""), classes_count.get("3",""), classes_count.get("4",""),
                                classes_perc.get("0",""), classes_perc.get("1",""), classes_perc.get("2",""), classes_perc.get("3",""), classes_perc.get("4",""),
                                p[0], r[0], f[0], s[0], 
                                p[1], r[1], f[1], s[1], 
                                p[2], r[2], f[2], s[2],
                                p[3], r[3], f[3], s[3], 
                                p[4], r[4], f[4], s[4],
                                af, ap, 
                                mp, mr, mf, ms, 
                                wp, wr, wf, ws                         
                            ] + metrics_list)

        def process_file_predict(num_seq, file_data, csv_writer):
            
            file_data.seek(0)
            lines = file_data.readlines()   

            k_fold = file_data.name.split("/")[2].split(".")[0]

            for start, line in enumerate(lines):
                if "image, value, true_label, predict_label" in line:
                    break
            
            jump = False
            for i in range(start+1, len(lines)):
                if jump:
                    jump = False
                else:
                    results = lines[i].replace("\n", "").replace(" ","").split(",")
                    if len(results)!=4:
                        new_line = f"{lines[i]}{lines[i+1]}"
                        results = new_line.replace("\n", "").replace(" ","").split(",")
                        jump=True
                    img = results[0].split("/")[2].split(".")[0]
                    if len(img)>33:
                        img = img[:33]
                    true_lbl = results[2]
                    predict_lbl = results[3]
                    if true_lbl==predict_lbl:
                        predict_res=True
                    else:
                        predict_res=False
                    csv_writer.writerow([
                                    num_seq, k_fold, img, true_lbl, predict_lbl, predict_res
                                    ])
    
        file_metadata = f"{folder_stp}/metadata.txt"
        try:
            for e in os.listdir(folder_res):
                if os.path.isfile(os.path.join(folder_res, e)):
                    file_data = open(f"{folder_res}/{e}")
                    process_file_result(num_seq, self.k, self.epochs, file_data, file_metadata, self.results)
                    process_file_predict(num_seq, file_data, self.predict)
        except Exception as exp:
            print(f"Error while processing experiment {num_seq} results: {exp}.")

