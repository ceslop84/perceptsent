import os
import time
import json
import csv

def process_file(id, k, epochs, file_data, file_metadata, csv_writer):

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
                        id, k, epochs,
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

def process_file_predict(id, file_data, csv_writer):
    
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
                            id, k_fold, img, true_lbl, predict_lbl, predict_res
                            ])
                       
def write_headers(csv_writer):

    csv_writer.writerow([
                        "id", "k", "epochs",
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
    
def write_headers_predict(csv_writer):
    csv_writer.writerow([
                        "id", "k_fold", "img", "true_label", "predict_label", "predict_result"
                        ])

def process_folder(id, k, epochs, folder_res, folder_stp, csv_writer, csv_writer_predict):
    id_m = id.split("_")[0]
    file_metadata = f"{folder_stp}/{id_m}/metadata.txt"
    folder_id = f"{folder_res}/{id}"
    try:
        for e in os.listdir(folder_id):
            if os.path.isfile(os.path.join(folder_id, e)) and (e!="test_heatmap.txt") and (e!="test_1_1.txt") :
                file_data = open(f"{folder_id}/{e}")
                process_file(id, k, epochs, file_data, file_metadata, csv_writer)
                process_file_predict(id, file_data, csv_writer_predict)
        print(f"Experimento {id} processado com sucesso!")
    except Exception as exp:
        print(f"Erro ao processar experimento {id}: {exp}.")

def quick_check(folder_res, file_stp, k):
    import re
    with open(file_stp, 'r') as e:
        lines = e.readlines()
        for line in lines:
            check = [False]*5
            exp = line.replace("\n", "").split(";")
            id = int(exp[0])
            for root, dirs, files in os.walk(f"{folder_res}/{id}"):
                for i in range(k):
                    regex = re.compile(fr"test_(.*)_{i+1}.txt")
                    for file in files:
                        res = re.match(regex, file)
                        if res:
                            check[i]=True
                            break
            for c in check:
                if c is False:
                    print (f"{id};ERRO")

if __name__ == '__main__':

    k = 5
    epochs = 20
    
    folder_stp = "images2/setup"
    file_stp = f"{folder_stp}/experiments.txt"
    folder_res = "output"
    file_res = f"{folder_res}/results.csv"
    file_res_predict = f"{folder_res}/results_predict.csv"

    csv_res = open(file_res, 'w+', newline='')
    writer_res = csv.writer(csv_res, delimiter=';')

    csv_res_predict = open(file_res_predict, 'w+', newline='')
    writer_res_predict = csv.writer(csv_res_predict, delimiter=';')

    write_headers(writer_res)
    write_headers_predict(writer_res_predict)
    
    # quick_check(folder_res, file_stp, k)

with open(file_stp, 'r') as e:
    lines = e.readlines()
    for line in lines:
        exp = line.replace("\n", "").split(";")
        id = str(exp[0])
        process_folder(id, k, epochs, folder_res, folder_stp, writer_res, writer_res_predict)

    #process_folder("281", k, epochs, folder_res, folder_stp, writer_res, writer_res_predict)
        
    csv_res.close()