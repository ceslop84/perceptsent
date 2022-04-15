import os
import csv
import json
from random import shuffle
import numpy as np
import pandas as pd
from copy import deepcopy

def export_data(file_in, folder_out, nr_classes, expand_neutral=None, shift_neutral=None):
    
    create_dir(folder_out)

    with open(file_in, 'r') as j:
        json_data = json.load(j)

    info_cd = get_coded_informations()
    info_lb = get_label_informations()
    sent_dict = get_classes_dict(nr_classes, None, expand_neutral, shift_neutral)
    sent_list = get_classes_list(nr_classes, shift_neutral)
    data_lb = list()
    data_cd = list()
    headers = ["worker", "assignment", "img", "sentiment", "in_out_door", "age", "gs", "eco", "edu", "opt1", "neg1", "opt2", "comments", "reasons"]
    data_lb.append(headers)
    data_cd.append(headers)
    
    workers = dict()
    for assig in json_data["tarefas"]:
        worker = assig["worker_id"]
        if not workers.get(worker, None):
            workers[worker] = assig["info_resp"]
    
    for assig in json_data["tarefas"]:
        resp = workers.get(assig["worker_id"])
        if resp is None:
            resp = dict()
            resp["age"] = "noage"
            resp["gs"] = "nogender"
            resp["eco"] = "noeco"
            resp["edu"] = "noedu"
            resp["opt1"] = "no"
            resp["neg1"] = "no"
            resp["opt2"] = "no"

        for img in assig["image_resps"]:

            comments = list()
            reasons_cd = [0] * 37
            reasons_lb = list()
            for r in img["reasons"]:
                r_id = info_cd["reasons"].get(r, -1)
                if r_id >0:
                    reasons_lb.append(r)
                    reasons_cd[r_id]+=1
                else:
                    r.replace(",", "")
                    comments.append(r)

            worker_id = assig["worker_id"]
            assig_id= assig["assignment_id"]
            img_id = img["id"]

            sent_cd = sent_dict.get(img["sentiment"])                      
            in_out_door_cd = info_cd["in_out_door"].get(img["in_out_door"])
            age_cd = info_cd["age"].get(resp["age"])
            gs_cd = info_cd["gs"].get(resp["gs"])
            eco_cd = info_cd["eco"].get(resp["eco"])
            edu_cd = info_cd["edu"].get(resp["edu"])
            opt1_cd = info_cd["opt1"].get(resp["opt1"])
            neg1_cd = info_cd["neg1"].get(resp["neg1"])
            opt2_cd = info_cd["opt2"].get(resp["opt2"])

            sent_lb = sent_list[sent_cd]                      
            in_out_door_lb = info_lb["in_out_door"].get(in_out_door_cd)
            age_lb = info_lb["age"].get(age_cd)
            gs_lb = info_lb["gs"].get(gs_cd)
            eco_lb = info_lb["eco"].get(eco_cd)
            edu_lb = info_lb["edu"].get(edu_cd)
            opt1_lb = info_lb["opt1"].get(opt1_cd)
            neg1_lb = info_lb["neg1"].get(neg1_cd)
            opt2_lb = info_lb["opt2"].get(opt2_cd)

            data_cd.append([worker_id, assig_id, img_id, sent_cd, in_out_door_cd, 
                                age_cd, gs_cd, eco_cd, edu_cd, opt1_cd, neg1_cd, opt2_cd,
                                ",".join(comments),
                                ",".join([str(i) for i in reasons_cd])])

            data_lb.append([worker_id, assig_id, img_id, sent_lb, in_out_door_lb, 
                                age_lb, gs_lb, eco_lb, edu_lb, opt1_lb, neg1_lb, opt2_lb,
                                ",".join(comments),
                                ",".join([str(i) for i in reasons_lb])])
    
    data_lb_t = list()
    data_lb_t.append(["assignment_img", "data", "field"])
    for dd in data_lb[1:]:
        for i, d in enumerate(dd[3:12]):
            data_lb_t.append([f"{dd[1]}_{dd[2]}", d, data_lb[0][i+3]])
        for r in dd[13].split(','):
            data_lb_t.append([f"{dd[1]}_{dd[2]}", r, "reasons"])
    
    data_cd_t = list()
    data_cd_t.append(["assignment_img", "data", "field"])
    for dd in data_cd[1:]:
        for i, d in enumerate(dd[3:12]):
            data_cd_t.append([f"{dd[1]}_{dd[2]}", d, data_cd[0][i+3]])
        for r in dd[13].split(','):
            data_cd_t.append([f"{dd[1]}_{dd[2]}", r, "reasons"])
    
    name = f"{nr_classes}c"
    if shift_neutral:
        name += f"_{shift_neutral[:3]}"
    if expand_neutral:
        name += "_exp"

    with open(f'{folder_out}/data_label_t_{name}.csv', 'w') as f:
        write = csv.writer(f, delimiter=';')
        for i in data_lb_t:
            write.writerow(i)
    
    with open(f'{folder_out}/data_code_t_{name}.csv', 'w') as f:
        write = csv.writer(f, delimiter=';')
        for i in data_cd_t:
            write.writerow(i)


    with open(f'{folder_out}/data_label_{name}.csv', 'w') as f:
        write = csv.writer(f, delimiter=';')
        for i in data_lb:
            write.writerow(i)

    with open(f'{folder_out}/data_code_{nr_classes}.csv', 'w') as f:
        write = csv.writer(f, delimiter=';')
        for i in data_cd:
            write.writerow(i)

def get_coded_informations():

    age = {
        'noage': 0,
        '19_25': 1, 
        '18_25': 1,
        '26_35': 2, 
        '36_45': 3, 
        'over45': 4, 
        }

    gs = {'nogender': 0, 'female': 1, 'male': 2}

    eco = {'noeco': 0, 'noedu': 0, 'low': 1, 'middle': 2,  'high': 3}

    edu = {
        'noedu': 0, 
        'lesssec': 1, 'ttv_edu': 1, 'sec_edu': 1, 
        'Bachelor’s degree': 2, 'bac_edu': 2,
        'Postgraduate': 3,  'pos_edu': 3
        }

    opt1 = {
        'no': 0, 'no_opt1': 0,
        'ta': 1, 'ta_opt1': 1,
        'pa': 2, 'pa_opt1': 2,
        'nt': 3, 'nt_opt1': 3,
        'pd': 4, 'pd_opt1': 4,
        'td': 5, 'td_opt1': 5
        }

    neg1 = {
        'no': 0, 'no_neg1': 0,
        'ta': 1, 'ta_neg1': 1,
        'pa': 2, 'pa_neg1': 2,
        'nt': 3, 'nt_neg1': 3,
        'pd': 4, 'pd_neg1': 4,
        'td': 5, 'td_neg1': 5
        }

    opt2 = {
        'no': 0, 'no_opt2': 0,
        'ta': 1, 'ta_opt2': 1,
        'pa': 2, 'pa_opt2': 2,
        'nt': 3, 'nt_opt2': 3,
        'pd': 4, 'pd_opt2': 4,
        'td': 5, 'td_opt2': 5
        }

    reasons = {
        "Accident": 0,
        "Animals": 1,
        "Art/Architecture": 2,
        "Bad weather": 3,
        "Celebration": 4,
        "Colors": 5,
        "Confusion": 6,
        "Debris/Destruction": 7,
        "Everyday image": 8,
        "Fire": 9,
        "Fireworks": 10,
        "Graffiti": 11,
        "Happiness/Smiles": 12,
        "Introspective": 13,
        "Lack of colors": 14,
        "Lack of Maintenance": 15,
        "Leisure/Fun/Rest": 16,
        "Low resolution/quality": 17,
        "Meaningless": 18,
        "Morbid": 19,
        "Nature": 20,
        "Negative text message": 21,
        "Night Lights": 22,
        "Organization": 23,
        "Panoramic view": 24,
        "Pleasant environment": 25,
        "Pollution": 26,
        "Positive text message": 27,
        "Poverty": 28,
        "Sky": 29,
        "Sports": 30,
        "Sunrise/Sunset": 31,
        "Tourist Attractions": 32,
        "Trash": 33,
        "Unpleasant colors": 34,
        "Violence": 35,
        'It has positive and negative elements (please indicate which in "Comments")': 36,
        }

    in_out_door = {'no_in_out_door': 0, 'Indoor': 1, 'Outdoor': 2}

    informations = {
        "age": age,
        "gs": gs,
        "eco": eco, 
        "edu": edu,
        "opt1": opt1,
        "neg1": neg1,
        "opt2": opt2,
        "reasons": reasons,
        "in_out_door": in_out_door
    }

    return informations

def get_label_informations():

    age = {
        0: 'no_age',
        1: '19_25',
        2: '26_35',
        3: '36_45', 
        4: 'over45' 
        }

    gs = {0: 'no_gender', 1: 'female', 2: 'male'}

    eco = {0: 'no_eco', 1: 'low', 2: 'middle', 3: 'high'}

    edu = {
        0: 'no_edu', 
        1: 'sec_edu', 
        2: 'bac_edu',
        3: 'pos_edu'
        }

    opt1 = {
        0: 'no_opt1',
        1: 'ta_opt1',
        2: 'pa_opt1',
        3: 'nt_opt1',
        4: 'pd_opt1',
        5: 'td_opt1'
        }

    neg1 = {
        0: 'no_neg1',
        1: 'ta_neg1',
        2: 'pa_neg1',
        3: 'nt_neg1',
        4: 'pd_neg1',
        5: 'td_neg1'
        }

    opt2 = {
        0: 'no_opt2',
        1: 'ta_opt2',
        2: 'pa_opt2',
        3: 'nt_opt2',
        4: 'pd_opt2',
        5: 'td_opt2'
        }

    reasons = {
        0: "Accident",
        1: "Animals",
        2: "Art/Architecture",
        3: "Bad weather",
        4: "Celebration",
        5: "Colors",
        6: "Confusion",
        7: "Debris/Destruction",
        8: "Everyday image",
        9: "Fire",
        10: "Fireworks",
        11: "Graffiti",
        12: "Happiness/Smiles",
        13: "Introspective",
        14: "Lack of colors",
        15: "Lack of Maintenance",
        16: "Leisure/Fun/Rest",
        17: "Low resolution/quality",
        18: "Meaningless",
        19: "Morbid",
        20: "Nature",
        21: "Negative text message",
        22: "Night Lights",
        23: "Organization",
        24: "Panoramic view",
        25: "Pleasant environment",
        26: "Pollution",
        27: "Positive text message",
        28: "Poverty",
        29: "Sky",
        30: "Sports",
        31: "Sunrise/Sunset",
        32: "Tourist Attractions",
        33: "Trash",
        34: "Unpleasant colors",
        35: "Violence",
        36: 'It has positive and negative elements (please indicate which in "Comments")',
        }

    in_out_door = {0: 'no_in_out_door',1: 'Indoor',2: 'Outdoor'}

    informations = {
        "age": age,
        "gs": gs,
        "eco": eco, 
        "edu": edu,
        "opt1": opt1,
        "neg1": neg1,
        "opt2": opt2,
        "reasons": reasons,
        "in_out_door": in_out_door
    }

    return informations

def get_info_resp(assgn, image, workers_info):
    worker_id = assgn.get('worker_id')
    worker_info = [x for x in workers_info if x["worker_id"] == worker_id and x["info_resp"]]
    if worker_info:
        data = worker_info[0].get("info_resp")
        data["reasons"] = image.get("reasons", [])
        return False, data
    else:
        return True, {
                       "age": "noage",
                       "gs": "nogender",
                       "eco": "noeco",
                       "edu": "noedu",
                       "opt1": "no_opt1",
                       "neg1": "no_neg1",
                       "opt2": "no_opt2",
                       "reasons": image.get("reasons", [])
                      }

def get_workers_info(tasks):
    workers_info = list()
    workers_list = list()
    keys=["worker_id", "info_resp"]
    for assgn in tasks:
        worker_id = assgn.get('worker_id')
        if not worker_id in workers_list:
            worker_info = [x for x in tasks if x["worker_id"] == worker_id and x["info_resp"]]
            if worker_info:
                workers_info.append(dict(zip(keys, list(map(worker_info[0].get, keys)))))
                workers_list.append(worker_id)
    return workers_info

def get_info_eval(classes, info_req, informations):
    info_eval = dict()
    for info_item in  info_req:
        info = informations.get(info_item)
        size= max(classes.values())+1
        cat_eval = dict()
        for cat_item in informations.get(info_item):
            cat_eval[cat_item] = [0] * size
        info_eval[info_item] = cat_eval
    return info_eval

def get_classes_count(nr_cls):
    if nr_cls==2:
        classes = {
            "0": 0,
            "1": 0
        }
    if nr_cls==3:
        classes = {
            "0": 0,
            "1": 0,
            "2": 0
        }
    if nr_cls==5:
        classes = {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0
        }
    return classes

def get_classes_list(nr_cls, shift_neutral=None):
    if nr_cls == 2:
        if shift_neutral=="positive":
            classes = ["Negative", "Other"]
        elif shift_neutral=="negative":
            classes = ["Other", "Positive"]
        else:
            raise Exception("Configuração inválida para definição de classes.")
    if nr_cls == 3:
        classes = ["Negative", "Neutral", "Positive"]
    if nr_cls == 5:
        classes = ["Negative", "SlightlyNegative", "Neutral", "SlightlyPositive", "Positive"]
    return classes

def get_classes_dict(nr_cls, method=None, expand_neutral=False, shift_neutral=None):
    
    if nr_cls == 5 or method=="average":
        classes = {
            "Negative": 0,
            "SlightlyNegative": 1,
            "Neutral": 2,
            "SlightlyPositive": 3,
            "Positive": 4
        }
    elif nr_cls == 2:
        if shift_neutral=="positive": 
            if expand_neutral:
                classes = {
                    "Negative": 0,
                    "SlightlyNegative": 1,
                    "Neutral": 1,
                    "SlightlyPositive": 1,
                    "Positive": 1
                }
            else:
                classes = {
                    "Negative": 0,
                    "SlightlyNegative": 0,
                    "Neutral": 1,
                    "SlightlyPositive": 1,
                    "Positive": 1
                }
        
        elif shift_neutral=="negative": 
            if expand_neutral:
                classes = {
                    "Negative": 0,
                    "SlightlyNegative": 0,
                    "Neutral": 0,
                    "SlightlyPositive": 0,
                    "Positive": 1
                } 
            else:
                classes = {
                    "Negative": 0,
                    "SlightlyNegative": 0,
                    "Neutral": 0,
                    "SlightlyPositive": 1,
                    "Positive": 1
                }   
        
        else:
            raise Exception("Error while processing the shift neutral parameter.")
    elif nr_cls == 3:
        if expand_neutral:
            classes = {
                "Negative": 0,
                "SlightlyNegative": 1,
                "Neutral": 1,
                "SlightlyPositive": 1,
                "Positive": 2
            }
        else:
            classes = {
                "Negative": 0,
                "SlightlyNegative": 0,
                "Neutral": 1,
                "SlightlyPositive": 2,
                "Positive": 2
            }
    else:
        raise Exception("Erro while processing the classes dictionary.")
    return classes

def report(imgs_list, imgs_summ, img_df, classes, cfg, folder_out, file_out, data_only=False):
    balance_count, balance_perc, balance_imgs = analyse_class_count(imgs_summ, classes)  
    metrics = build_metrics(img_df, cfg["nr_classes"])
    stats = build_stats(imgs_list, cfg["descriptors"], classes)
    classes_list = get_classes_list(cfg["nr_classes"], cfg["shift_neutral"])
    if data_only:
        with open(f"{folder_out}/{file_out}", 'w') as f:
            f.write(f"{json.dumps(cfg)}")
            f.write("\n")
            f.write(f"{','.join(classes_list)}")
            f.write("\n")
            f.write(f"{json.dumps(classes)}")
            f.write("\n")
            f.write(f"{json.dumps(balance_count)}")
            f.write("\n")
            f.write(f"{json.dumps(balance_perc)}")
            f.write("\n")
            f.write(f"{json.dumps(metrics)}")
            f.write("\n")
            f.write(f"{json.dumps(stats)}")
    else:
        with open(f"{folder_out}/{file_out}", 'w') as f:
            f.write(f"Configuration: {json.dumps(cfg, indent=2, default=str)}")
            f.write("\n\n")
            f.write(f"Classes (out): {json.dumps(classes_list, indent=2, default=str)}")
            f.write("\n\n")
            f.write(f"Classes (in): {json.dumps(classes, indent=2, default=str)}")
            f.write("\n\n")
            f.write(f"Metrics: {json.dumps(metrics, indent=2, default=str)}")
            f.write("\n\n")
            f.write(f"Balance count: {json.dumps(balance_count, indent=2, default=str)}")
            f.write("\n\n")
            f.write(f"Balance %: {json.dumps(balance_perc, indent=2, default=str)}")
            f.write("\n\n")
            f.write(f"Descriptors statistics: {json.dumps(stats, indent=2, default=str)}")

def find_value(dict, target):
    for key, value in dict.items(): 
        if value == target:
            return key

def build_stats(imgs, descriptors, classes):

    def calculate_stat(img, stats, cat, cat_id, eval_info, step=1):
        cat_label = find_value(cat, cat_id)
        cat_info = eval_info[cat_label]
        pos = img[2]
        cat_info[pos] += step
        stats[img[0]] = eval_img
    
    if not descriptors:
        return None

    stats = dict()
    informations = get_coded_informations()
    for img in imgs:
        eval_img = stats.get(img[0])
        if not eval_img:
            eval_img = get_info_eval(classes, descriptors, informations)  
        
        for i, desc in enumerate(descriptors):
            eval_info = eval_img.get(desc)
            cat = informations.get(desc)
            if desc=="reasons":
                reasons = img[3+i:]
                for cat_id, r in enumerate(reasons):
                    calculate_stat(img, stats, cat, cat_id, eval_info, r)
            else:
                calculate_stat(img, stats, cat, img[3+i], eval_info)
    return stats

def build_metrics(df, nr_classes, dict_fmt=True):

    def fleiss(df):
        # Calculate Fleiss.
        m = 5
        n = len(df.index)
        q = list()
        pe = 0.0
        for c in range(nr_classes):
            q_aux = df[f"sent_{c}"].sum()/(n*m)
            pe += pow(q_aux, 2)
            q.append(q_aux)
        sent_sum_quad = df["sent_sum_quad"].sum()
        pa = (sent_sum_quad-(n*m))/(n*m*(m-1))
        fleiss = (pa-pe)/(1-pe)
        return fleiss

    def cronbach (df):
        # Calculate Cronbach
        k = 5
        var_j = df["jdg_sum"].var()
        sum_var = 0.0
        var_jn = list()
        for j in range(5):
            var_jn.append(df[f"jdg_{j+1}"].var())
        sum_var = sum(var_jn)
        cronbach = (k/(k-1))*(1-(sum_var/var_j))
        return cronbach

    metrics = list()
    metrics.append(df["img_agr"].mean())
    metrics.append(df["img_dis"].mean())
    metrics.append(df["avg_img_wkr_agr"].mean())
    metrics.append(df["avg_img_wkr_dis"].mean())
    metrics.append(fleiss(df))
    metrics.append(cronbach(df))
    metrics.append(len(df.index))
    for c in range(nr_classes):
        metrics.append(len(df.loc[df["sent"] == c].index))    
    
    if dict_fmt:
        columns = ["img_agr", "img_dis", "avg_img_wkr_agr", "avg_img_wkr_dis", "fless", "cronbach", "imgs"]
        for c in range(nr_classes):
            columns.append(f"imgs_{c}")
        metrics_df = pd.DataFrame([metrics], columns=columns)
        return metrics_df.to_dict()
    else:
        return metrics

def build_metrics_profile(df_list, nr_classes, file_name=None):

    metrics = list()
    max_len = 0
    for c in range(nr_classes):
        curr_len = len(df_list[c].index)
        if curr_len > max_len:
            max_len = curr_len
    for w in range(0,max_len,10):
        head_list = list()
        for c in range(nr_classes):
            head_list.append(df_list[c].head(w))
        df = pd.concat(head_list)
        metrics.append(build_metrics(df, nr_classes, False))
    columns = ["img_agr", "img_dis", "avg_img_wkr_agr", "avg_img_wkr_dis", "fless", "cronbach", "imgs"]
    for c in range(nr_classes):
        columns.append(f"imgs_{c}")
    metrics_df = pd.DataFrame(metrics, columns=columns)
    metrics_df.to_csv(f"{file_name}.csv")

def extract_data(file_in, descriptors, classes, include_no_data):
    tasks = read_data(file_in)
    informations = get_coded_informations()
    workers_info = get_workers_info(tasks)
    imgs = list()
    
    for assgn in tasks:
        image_resps = assgn.get('image_resps')  
        for image in image_resps:
            sent = list()
            add_data = True
            sent.append(image.get('id'))
            sent.append(assgn.get('assignment_id'))
            sent.append(classes.get(image.get('sentiment')))

            if descriptors:
                no_data, info_resp = get_info_resp(assgn, image, workers_info)
                if not no_data or (include_no_data and no_data):                      
                    for desc in descriptors:
                        if desc == "reasons":
                            reasons = [0] * 37
                            for r in image.get("reasons", []):
                                r_id = informations["reasons"].get(r, -1)
                                if r_id >0:
                                    reasons[r_id]+=1
                            for r in reasons: sent.append(r)
                        else:
                            cat = info_resp.get(desc)
                            reasons = informations.get(desc).get(cat)
                            sent.append(reasons)               
                else:
                    add_data = False  
            
            if add_data:
                imgs.append(sent)    

    return imgs

def filter_data(imgs, descriptors, classes):
    stats = build_stats(imgs, descriptors, classes)
    imgs_del = list()
    for img_name, img_data  in stats.items():
        for cat_label, cat_data in img_data.items():
            for cls_label, cls_data in cat_data.items():
                zeros = cls_data.count(0)
                size = len(cls_data)
                if size - zeros > 1 and img_name not in imgs_del:
                    imgs_del.append(img_name)
    if imgs_del:
        imgs_filter = [img for img in imgs if img[0] not in imgs_del]
        return imgs_filter
    else:
        return imgs
                    
def read_data(file_in):
    with open(file_in, 'r') as j:
        json_data = json.load(j)
    tasks = json_data.get("tarefas")
    return tasks

def save_list(img_list, img_summ, folder, method, data_augmented, file):

    if data_augmented and method!="multi":

        img_summ_i = [i[0] for i in img_summ]
        img_summ_2 = list()
        for i in img_list:
            if i[0] in img_summ_i:
                index = img_summ_i.index(i[0])
                img_summ_2.append([i[0], i[1], img_summ[index][2]])
        img_summ = img_summ_2

    if method == "multi" or data_augmented:
        names = list()
        workers = list()
        for img in img_summ:
            if img[0] not in names:
                names.append(img[0])
                workers.append(f"{img[1]} {str(img[2])}")
            else:
                index = names.index(img[0])
                workers[index] += f" {img[1]} {str(img[2])}"

        with open(f"{folder}/{file}", 'w') as f:
            for img in zip(names, workers):
                f.write(f"{img[0]} {img[1]}\n")
    else:
        with open(f"{folder}/{file}", 'w') as f:
            for img in img_summ:
                f.write(f"{img[0]} {str(img[2])}\n")

def save_descriptors(imgs, folder, method, descriptors, data_augmented):
    if not descriptors:
        return None

    file_names = list()
    desc_content = list()
    
    if method == "multi":

        for img in imgs:
            file_names.append(f"{img[0]}_{img[1]}.txt")
            desc_content.append(img[3:])
    else:

        workers_treat = list()
        imgs_treat = list()

        informations = get_coded_informations()
        desc_template = list()
        for i, desc_label in enumerate(descriptors):
            desc = informations.get(desc_label)
            class_max = max(desc, key=desc.get)
            tam = desc.get(class_max)
            desc_list = [0] * (tam+1)
            desc_template.append(desc_list)

        for img in imgs:
            
            if img[0] not in imgs_treat:
                imgs_treat.append(img[0])
                workers_treat.append(list())
                desc_content.append(deepcopy(desc_template))
            img_id = imgs_treat.index(img[0])   
            workers_treat[img_id].append(img[1])            
            for i, desc_label in enumerate(descriptors):
                class_id = img[3+i]
                desc_content[img_id][i][class_id] += 1
        
        if data_augmented:
            desc_content_2 = list()
            for i, img in enumerate(imgs_treat):
                for wkr in workers_treat[i]:
                    file_names.append(f"{img}_{wkr}.txt")
                    desc_content_2.append(desc_content[i])
            desc_content = desc_content_2
        else:
            for i, img in enumerate(imgs_treat):
                file_names.append(f"{img}.txt")
        

    for file_name, data in zip(file_names, desc_content):
        check = False
        if len(data) > 1:
            if isinstance(data[0], list):
                file_content = [i for item in data for i in item]
            else:
                check = True
        else:
            check = True
        if check:
            file_content = data
        np.savetxt(f"{folder}/{file_name}", np.asarray(file_content, dtype="float32")) 

def majority(imgs, nr_classes, threshold):
    names = list()
    votes = list()
    for img in imgs:
        if img[0] not in names:
            names.append(img[0])
            vote = [0] * nr_classes
            vote[img[2]] += 1
            votes.append(vote)
        else:
            index = names.index(img[0])
            votes[index][img[2]] += 1
    imgs = list()
    for img in zip(names, votes):
        for sent, votes in enumerate(img[1]):
            if votes >= threshold:
                imgs.append([img[0], None, sent])
                break
    return imgs

def average(imgs, nr_classes):
    if nr_classes == 2:
        scale = [0, 2, 5]
    elif nr_classes == 3:
        scale = [0, 1.2, 2.8, 5]
    elif nr_classes == 5:
        scale = [0, 1, 2, 3, 4, 5]
    names = list()
    votes = list()
    ids = list()
    i=-1
    for img in imgs:
        i += 1
        if img[0] not in names:
            names.append(img[0])
            vote = [0] * 5
            vote[img[2]] += 1
            votes.append(vote)
            id = list()
            id.append(i)
            ids.append(id)
        else:
            index = names.index(img[0])
            votes[index][img[2]] += 1
            ids[index].append(i)
    imgs2 = list()
    for img in zip(names, votes, ids):
        sum_sent = 0
        sum_votes = 0
        for sent, votes in enumerate(img[1]):
            sum_sent += (sent) * votes
            sum_votes += votes
        avg = sum_sent / sum_votes
        for sent, meas in enumerate(scale):
            if avg>=scale[sent] and avg<scale[sent+1]:
                sent_out = sent
                imgs2.append([img[0], None, sent_out])
                break
        for id in img[2]:
            imgs[id][2] = sent_out
    return imgs2

def summarize(imgs, method, nr_classes):
    if method == "majority1":
        imgs_summ = majority(imgs, nr_classes, 1)
    if method == "majority2":
        imgs_summ = majority(imgs, nr_classes, 2)
    if method == "majority3":
        imgs_summ = majority(imgs, nr_classes, 3)
    if method == "majority4":
        imgs_summ = majority(imgs, nr_classes, 4)
    if method == "majority5":
        imgs_summ = majority(imgs, nr_classes, 5)
    if method == "average":
        imgs_summ = average(imgs, nr_classes)
    if method == "multi":
        imgs_summ = imgs
    imgs_summ_np = np.asarray(imgs_summ)
    imgs_summ_name = imgs_summ_np[:,0]
    imgs_list = [img for img in imgs if img[0] in imgs_summ_name]
    return imgs_list, imgs_summ

def analyse_class_count(imgs, classes):
    nr_classes = classes.get(max(classes, key=classes.get))+1
    balance_count = get_classes_count(nr_classes)
    for img in imgs:
        balance_count[f"{img[2]}"] += 1
    min = -1
    cls_min = ""
    for cls_label, cls_count in balance_count.items():
        if min<0:
            min = cls_count
            cls_min = cls_label
        elif cls_count<min:
            min = cls_count
            cls_min = cls_label
    balance_imgs = dict()
    balance_perc = dict()
    for cls_label, cls_count in balance_count.items():
        balance_imgs[cls_label] = [i for i in imgs if i[2]==int(cls_label)]
        balance_perc[cls_label] = balance_count[cls_label]/balance_count[cls_min]
    return balance_count, balance_perc, balance_imgs

def calibrate(imgs_list, imgs_summ, method, class_balance, classes):
    assert (method!="multi"), "The method multi doesn't allow to apply class balance calibration."
    imgs_summ_bal = list()
    balance_count, balance_perc, balance_imgs = analyse_class_count(imgs_summ, classes)    
    for cls_label, cls_perc in balance_perc.items():
        if cls_perc > class_balance:
            red_coef = class_balance/cls_perc
            reduction = balance_count.get(cls_label) - (balance_count.get(cls_label) * red_coef)
            imgs_cls = balance_imgs.get(cls_label)
            while reduction > 0:
                shuffle(imgs_cls)
                item = imgs_cls.pop(0)
                reduction -= 1
        imgs_summ_bal += balance_imgs[cls_label]  
    imgs_list_bal = list()
    for img in imgs_summ_bal:
        imgs_list_bal += [i for i in imgs_list if i[0]==img[0]]
    return imgs_list_bal, imgs_summ_bal

def config(nr_classes, expand_neutral, shift_neutral, method, 
            data_augmented, metadata, semantic, filter, 
            dataset_strategy, dataset_size,
            class_balance, include_no_data):
    cfg = dict()
    cfg["method"] = method
    cfg["nr_classes"] = nr_classes
    cfg["expand_neutral"] = expand_neutral
    cfg["shift_neutral"] = shift_neutral
    cfg["data_augmented"] = data_augmented

    descriptors = list()
    if metadata:
        descriptors += ["age", "gs", "eco", "edu", "opt1", "neg1", "opt2"]
    if semantic:
        descriptors += ["reasons"]
    cfg["descriptors"] = descriptors

    cfg["filter"] = filter
    cfg["class_balance"] = class_balance
    cfg["include_no_data"] = include_no_data
    cfg["dataset_strategy"] = dataset_strategy
    cfg["dataset_size"] = dataset_size
    return cfg

def create_dir(dir):
    if not os.path.isdir(dir):
            try:
                os.mkdir(dir)
                return dir
            except Exception as e:
                print(f"Creation of directory {dir} failed: {e}")
                exit(1)
    else:
        return dir

def create_dataset(img_list, img_summ, img_data, nr_classes, 
                   dataset_strategy, dataset_size,
                   dataset_profiling, folder_out, file_name):
    
    def process_dataset(img_data, img_summ):

        # Create index for the sentiments summarized.
        img_summ_index = [img[0] for img in img_summ]

        # Calculating stats.
        img_index = list()
        img_votes = list()
        img_jdg = list()
        wkr_index = list()
        wkr_sent = list()
        for d in img_data:
            img = d[0]
            wkr = d[1]
            sent = d[2]
            wkr_img = wkr+" "+img
            if img not in img_index:
                img_index.append(img) 
                img_votes.append([0] * nr_classes)
                img_jdg.append([None] * 5)
            if wkr_img not in wkr_index:
                wkr_index.append(wkr_img) 
                wkr_sent.append(sent)
            img_votes[img_index.index(img)][sent] += 1
            img_jdg[img_index.index(img)][sum(img_votes[img_index.index(img)])-1] = sent
        del d, img, wkr, sent, wkr_img, 
        
        # Statistics convolution.
        img_votes_conv = list()
        for v in img_votes:
            conv = list()
            for index_main, item_main in enumerate(v):
                if item_main == 0:
                    conv.append(0)
                else:
                    sum_item = 0
                    for index_conv, item_conv in enumerate(v):
                        sum_item += item_conv*abs(index_main-index_conv)
                    conv.append(sum_item/item_main)
            img_votes_conv.append(conv)  
        del conv, v, index_main, item_main, index_conv, item_conv, sum_item

        # Calculating image agreement/disagreement.
        img_agr = list()
        img_dis = list()
        for index, v in enumerate(img_votes):
            img_agr.append(round(max(v)/sum(v),2))
            img_dis.append(sum(img_votes_conv[index]))
        # Min-Max Scaling.
        img_max_dis = max(img_dis)
        img_dis = [round(i/img_max_dis,2) for i in img_dis]
        del img_max_dis, index, v

        # Calculating worker agreement/disagreement.
        wkr_agr = list()
        wkr_dis = list()
        for index, w in enumerate(wkr_index):
            wkr_img = w.split(" ")
            wkr = wkr_img[0]
            img = wkr_img[1]
            stat = img_votes[img_index.index(img)]
            sent = wkr_sent[index]
            common_raters = stat[sent]
            wkr_agr.append(round((common_raters-1)/4,2))
            wkr_dis.append(img_votes_conv[img_index.index(img)][sent]/stat[sent])
        # Min-Max Scaling.
        wkr_max_dis = max(wkr_dis)
        wkr_dis = [round(i/wkr_max_dis,2) for i in wkr_dis]
        del wkr_max_dis, common_raters, index, wkr_img, w, wkr, img, stat, sent

        # Consolidate image statistics...
        img_stat = list()
        for i in zip(img_index, img_agr, img_dis, img_votes, img_jdg):
            values = [i[0], i[1], i[2]]
            # Processing votes.
            sent_sum_quad = 0
            for v in i[3]:
                sent_sum_quad += pow(v, 2)
                values.append(v)
            values.append(sent_sum_quad)
            # Processing judges.
            jdg_sum = 0
            for j in i[4]:
                jdg_sum += j
                values.append(j)
            values.append(jdg_sum)
            # Find the summarized sentiment, accordingly to the chosen method.
            if i[0] in img_summ_index:
                values.append(img_summ[img_summ_index.index(i[0])][2])
            img_stat.append(values)
        columns = ["img", "img_agr", "img_dis"]
        for c in range(nr_classes):
            columns.append(f"sent_{c}")
        columns += ["sent_sum_quad", "jdg_1", "jdg_2", "jdg_3", "jdg_4", "jdg_5", "jdg_sum", "sent"]
        img_stat_df = pd.DataFrame(img_stat, columns=columns)
        del i, v, j, c, jdg_sum, sent_sum_quad, columns, values, img_stat, img_jdg
        
        # Consolidate worker statistics...
        wkr_stat = list()
        for w in zip (wkr_index, wkr_sent, wkr_agr, wkr_dis):
            wkr_img = w[0].split(" ")
            wkr = wkr_img[0]
            img = wkr_img[1]
            values = [wkr, img, w[1], w[2], w[3]]
            # Processing votes.
            votes = img_votes[img_index.index(img)]
            for v in votes:
                values.append(v)
            # Find the summarized sentiment, accordingly to the chosen method.
            if img in img_summ_index:
                values.append(img_summ[img_summ_index.index(img)][2])
            wkr_stat.append(values)
        columns = ["wkr", "img", "wkr_sent", "wkr_agr", "wkr_dis"]
        for c in range(nr_classes):
            columns.append(f"sent_{c}")
        columns.append("sent")
        wkr_stat_df = pd.DataFrame(wkr_stat, columns=columns)
        del w, c, v, votes, columns, values, wkr_stat, wkr, img
        del img_index, img_agr, img_dis, img_votes, img_votes_conv, wkr_index, wkr_agr, wkr_dis, wkr_sent, wkr_img

        # Calculating worker agr/dis average, considering the image average.
        avg_wkr_agr = list()
        avg_wkr_dis = list()
        wkrs = list()
        for index, row in img_stat_df.iterrows():
            i_wkrs = wkr_stat_df.loc[wkr_stat_df["img"] == row["img"]]
            avg_wkr_agr.append(i_wkrs["wkr_agr"].mean())
            avg_wkr_dis.append(i_wkrs["wkr_dis"].mean())
            wkrs.append(" ".join(i_wkrs["wkr"].tolist()))
        img_stat_df["avg_img_wkr_agr"] = avg_wkr_agr
        img_stat_df["avg_img_wkr_dis"] = avg_wkr_dis
        img_stat_df["wkrs"] = wkrs
        del avg_wkr_agr, avg_wkr_dis, wkrs, i_wkrs, index, row

        # Calculating worker agr/dis average, considering the worker average.
        wkr_stat_group_df = wkr_stat_df.groupby(["wkr"], as_index=False).mean()
        wkr_stat_group_df.drop(["wkr_sent", "sent"], axis='columns', inplace=True)
        for c in range(nr_classes):
            wkr_stat_group_df.drop([f"sent_{c}"], axis='columns', inplace=True)
        wkr_stat_group_df = wkr_stat_group_df.sort_values(by=["wkr_agr", "wkr_dis", "wkr"], ascending=[True, False, True])
        del wkr_stat_df

        return img_stat_df, wkr_stat_group_df

    def img_agr_dataset(img_stat_df):
        # Creating the datasets - Image quality
        df = img_stat_df.sort_values(by=["img_agr", "img_dis", "img"], ascending=[False, True, True])
        img_agr = list()
        for c in range(nr_classes):
            img_agr.append(df.loc[df["sent"] == c])
        del df, c
        return img_agr
    
    def wkr_agr_dataset(img_stat_df, wkr_stat_df):
        # Creating the datasets - Worker quality (wkr average)
        imgs = list()
        for w in range(2500,-5,-5):
            w_wkrs = wkr_stat_df.head(w)
            filter_wkrs = "|".join(list(w_wkrs["wkr"]))
            if filter_wkrs:
                w_img = img_stat_df[img_stat_df['wkrs'].str.contains(filter_wkrs)==False]
            else:
                w_img = img_stat_df
            w_img = w_img["img"].values.tolist()
            if len(imgs)>0 and (len(w_img) > len(imgs)):
                for i in w_img:
                    if i not in imgs:
                        imgs.append(i)
            elif len(imgs)==0 and len(w_img)>0:
                imgs += w_img
        imgs_full = list()
        for i in imgs:
            imgs_full += img_stat_df.loc[img_stat_df["img"] == i].values.tolist()
        columns = ["img", "img_agr", "img_dis"]
        for c in range(nr_classes):
            columns.append(f"sent_{c}")
        columns += ["sent_sum_quad", "jdg_1", "jdg_2", "jdg_3", "jdg_4", "jdg_5", "jdg_sum", "sent", "avg_img_wkr_agr", "avg_img_wkr_dis", "wkrs"]
        df = pd.DataFrame(imgs_full, columns=columns)
        wkr_agr = list()
        for c in range(nr_classes):
            wkr_agr.append(df.loc[df["sent"] == c])
        del imgs, w_wkrs, filter_wkrs, w_img, w, columns, df, c

        return wkr_agr

    img_stat_df, wkr_stat_df = process_dataset(img_data, img_summ)

    if dataset_strategy in ["img_agr",""]:
        df_list = img_agr_dataset(img_stat_df)
    elif dataset_strategy == "wkr_agr":
        df_list = wkr_agr_dataset(img_stat_df, wkr_stat_df)

    if dataset_profiling:
        build_metrics_profile(df_list, nr_classes, f"{folder_out}/{file_name}")
    
    df_list_head = list()
    for c in range(nr_classes):
        df_list_head.append(df_list[c].head(dataset_size))
    img_df = pd.concat(df_list_head)

    img_index = img_df["img"].values.tolist()
    img_summ = [img for img in img_summ if img[0] in img_index]
    img_list = [img for img in img_list if img[0] in img_index]

    return img_df, img_list, img_summ

def generate_database(file_in, folder_out, cfg):
    """
    nr_classes = [5, 3, 5]
    expand_neutral = [True, False, None]
    shift_neutral = ["positive", "negative", None]
    method = ["multi", "majority_3", "majority_4", "majority_5", "average"]
    data_augmented = [True, False, None]
    descriptors = [None, "age", "gs", "eco", "edu", "opt1", "neg1", "opt2", "reasons"]
    filter = [True, False]
    class_balance = [0, 1.25] (p = samples from greater class/samples from smaller class, i.e.: 100/10 = 10.0)
    include_no_data = [False, True]
    dataset_strategy = [img_agr, wkr_agr]
    dataset_size = [0, ..., 5000] (integer value)
    
    """
    dataset_profiling = False
    classes = get_classes_dict(cfg["nr_classes"], cfg["method"], cfg["expand_neutral"], cfg["shift_neutral"])    
    img_data = extract_data(file_in, cfg["descriptors"], classes, cfg["include_no_data"])
    if cfg["filter"]:
        img_data = filter_data(img_data, cfg["descriptors"], classes)
    img_list, img_summ = summarize(img_data, cfg["method"], cfg["nr_classes"])
    classes = get_classes_dict(cfg["nr_classes"], None, cfg["expand_neutral"], cfg["shift_neutral"])  
    if cfg["class_balance"]:
        img_list, img_summ = calibrate(img_list, img_summ, cfg["method"], cfg["class_balance"], classes)
    
    img_df, img_list, img_summ = create_dataset(img_list, img_summ, img_data, cfg["nr_classes"], cfg["dataset_strategy"], cfg["dataset_size"], dataset_profiling, folder_out, "profil.csv")
    report(img_list, img_summ, img_df, classes, cfg, folder_out, f"metadata.txt", True)
    report(img_list, img_summ, img_df, classes, cfg, folder_out, f"report.txt", False)
    
    save_list(img_list, img_summ, folder_out, cfg["method"], cfg["data_augmented"], "data.txt")
    save_descriptors(img_list, folder_out, cfg["method"], cfg["descriptors"], cfg["data_augmented"])
    export_data(file_in, folder_out, cfg["nr_classes"], cfg["expand_neutral"], cfg["shift_neutral"])  


    
