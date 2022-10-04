import os
import csv
import json
from random import shuffle
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

class Dataset():
        
    def __init__(self, cfg, dataset, output, profiling):
        """
        nr_classes = [5, 3, 5]
        expand_neutral = [True, False, None]
        shift_neutral = ["positive", "negative", None]
        method = ["individual", "dominant_3", "dominant_4", "dominant_5", "average"]
        data_augmented = [True, False, None]
        metadatada = ["individual", "aggregated"]
        perceptions = ["individual", "aggregated"]
        descriptors = [None, "age", "gs", "eco", "edu", "opt1", "neg1", "opt2", "perceptions", "objects"]
        filter = [True, False]
        class_balance = [0, 1.25] (p = samples from greater class/samples from smaller class, i.e.: 100/10 = 10.0)
        dataset_strategy = [img_agr, wkr_agr]
        dataset_size = [0, ..., 5000] (integer value)
        
        """

        def create_coded_informations():

            age = {
                'no_age': 0,
                '19_25': 1, 
                '18_25': 1,
                '26_35': 2, 
                '36_45': 3, 
                'over45': 4, 
                }

            gs = {'no_gender': 0, 'female': 1, 'male': 2}

            eco = {'no_eco': 0, 'low': 1, 'middle': 2,  'high': 3}

            edu = {'no_edu': 0, 
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

            perceptions = {
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

            objects = {
                "ambulance": 0,
                "anchor": 1,
                "animal": 2,
                "anvil": 3,
                "atv": 4,
                "ball": 5,
                "baloon": 6,
                "barbed wire": 7,
                "barrel": 8,
                "barrell": 9,
                "beach": 10,
                "bed": 11,
                "bench": 12,
                "bicicle": 13,
                "birdhouse": 14,
                "boat": 15,
                "bomb": 16,
                "bones": 17,
                "bookshelve": 18,
                "bookstand": 19,
                "bridge": 20,
                "building": 21,
                "bus": 22,
                "cactus": 23,
                "camera": 24,
                "car": 25,
                "carriage": 26,
                "cave": 27,
                "chair": 28,
                "chemical toilet": 29,
                "chimney": 30,
                "clock": 31,
                "coffe machine": 32,
                "collumn": 33,
                "container": 34,
                "couch": 35,
                "court": 36,
                "crane": 37,
                "crosswalk": 38,
                "debris": 39,
                "dirt": 40,
                "door": 41,
                "dumpster": 42,
                "excavator": 43,
                "fan": 44,
                "ferris wheel": 45,
                "fire": 46,
                "firefighter": 47,
                "fireplace": 48,
                "firetruck": 49,
                "fireworks": 50,
                "flag": 51,
                "flowers": 52,
                "food stand": 53,
                "fork": 54,
                "fountain": 55,
                "gas pump": 56,
                "gate": 57,
                "glove": 58,
                "graffiti": 59,
                "grass": 60,
                "grave": 61,
                "guitar": 62,
                "hammock": 63,
                "helicopter": 64,
                "ice": 65,
                "insect": 66,
                "key": 67,
                "lava": 68,
                "lawnmower": 69,
                "light pole": 70,
                "lock": 71,
                "mail box": 72,
                "manhole cover": 73,
                "marshmallow": 74,
                "mattress": 75,
                "megaphone": 76,
                "moon": 77,
                "motor": 78,
                "motorcycle": 79,
                "mountain": 80,
                "oven": 81,
                "pathway": 82,
                "pen": 83,
                "person": 84,
                "phone": 85,
                "phone booth": 86,
                "pizza": 87,
                "plane": 88,
                "police car": 89,
                "policeman": 90,
                "pool": 91,
                "power pole": 92,
                "power wires": 93,
                "pushcart": 94,
                "pyramid": 95,
                "radio": 96,
                "rainbow": 97,
                "river": 98,
                "road": 99,
                "rocks": 100,
                "sea": 101,
                "shoe": 102,
                "shopping cart": 103,
                "sidewalk": 104,
                "sign": 105,
                "skateboard": 106,
                "sky": 107,
                "smoke": 108,
                "soldier": 109,
                "spoon": 110,
                "statue": 111,
                "street": 112,
                "stuffed animal": 113,
                "sun": 114,
                "syringe": 115,
                "table": 116,
                "tank": 117,
                "tent": 118,
                "toilet": 119,
                "tractor": 120,
                "traffic cone": 121,
                "trailer home": 122,
                "train": 123,
                "train track": 124,
                "trash": 125,
                "trash can": 126,
                "tree": 127,
                "tree trunk": 128,
                "truck": 129,
                "tunnel": 130,
                "umbrella": 131,
                "van": 132,
                "vegetation": 133,
                "vr": 134,
                "wall": 135,
                "water tank": 136,
                "waterfall": 137,
                "weapom": 138,
                "weapon": 139,
                "wheel": 140,
                "wheelchair": 141,
                "wind turbine": 142,
                "windmill": 143,
                "window": 144,
                "wristwatch": 145
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
                "perceptions": perceptions,
                "objects": objects,
                "in_out_door": in_out_door
            }

            return informations

        def create_label_informations():

            age = {0: 'no_age', 1: '19_25', 2: '26_35', 3: '36_45', 4: 'over45'}

            gs = {0: 'no_gender', 1: 'female', 2: 'male'}

            eco = {0: 'no_eco', 1: 'low', 2: 'middle', 3: 'high'}

            edu = {0: 'no_edu', 1: 'sec_edu', 2: 'bac_edu',3: 'pos_edu'}

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

            perceptions = {
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

            objects = {
                0: "ambulance",
                1: "anchor",
                2: "animal",
                3: "anvil",
                4: "atv",
                5: "ball",
                6: "baloon",
                7: "barbed wire",
                8: "barrel",
                9: "barrell",
                10: "beach",
                11: "bed",
                12: "bench",
                13: "bicicle",
                14: "birdhouse",
                15: "boat",
                16: "bomb",
                17: "bones",
                18: "bookshelve",
                19: "bookstand",
                20: "bridge",
                21: "building",
                22: "bus",
                23: "cactus",
                24: "camera",
                25: "car",
                26: "carriage",
                27: "cave",
                28: "chair",
                29: "chemical toilet",
                30: "chimney",
                31: "clock",
                32: "coffe machine",
                33: "collumn",
                34: "container",
                35: "couch",
                36: "court",
                37: "crane",
                38: "crosswalk",
                39: "debris",
                40: "dirt",
                41: "door",
                42: "dumpster",
                43: "excavator",
                44: "fan",
                45: "ferris wheel",
                46: "fire",
                47: "firefighter",
                48: "fireplace",
                49: "firetruck",
                50: "fireworks",
                51: "flag",
                52: "flowers",
                53: "food stand",
                54: "fork",
                55: "fountain",
                56: "gas pump",
                57: "gate",
                58: "glove",
                59: "graffiti",
                60: "grass",
                61: "grave",
                62: "guitar",
                63: "hammock",
                64: "helicopter",
                65: "ice",
                66: "insect",
                67: "key",
                68: "lava",
                69: "lawnmower",
                70: "light pole",
                71: "lock",
                72: "mail box",
                73: "manhole cover",
                74: "marshmallow",
                75: "mattress",
                76: "megaphone",
                77: "moon",
                78: "motor",
                79: "motorcycle",
                80: "mountain",
                81: "oven",
                82: "pathway",
                83: "pen",
                84: "person",
                85: "phone",
                86: "phone booth",
                87: "pizza",
                88: "plane",
                89: "police car",
                90: "policeman",
                91: "pool",
                92: "power pole",
                93: "power wires",
                94: "pushcart",
                95: "pyramid",
                96: "radio",
                97: "rainbow",
                98: "river",
                99: "road",
                100: "rocks",
                101: "sea",
                102: "shoe",
                103: "shopping cart",
                104: "sidewalk",
                105: "sign",
                106: "skateboard",
                107: "sky",
                108: "smoke",
                109: "soldier",
                110: "spoon",
                111: "statue",
                112: "street",
                113: "stuffed animal",
                114: "sun",
                115: "syringe",
                116: "table",
                117: "tank",
                118: "tent",
                119: "toilet",
                120: "tractor",
                121: "traffic cone",
                122: "trailer home",
                123: "train",
                124: "train track",
                125: "trash",
                126: "trash can",
                127: "tree",
                128: "tree trunk",
                129: "truck",
                130: "tunnel",
                131: "umbrella",
                132: "van",
                133: "vegetation",
                134: "vr",
                135: "wall",
                136: "water tank",
                137: "waterfall",
                138: "weapom",
                139: "weapon",
                140: "wheel",
                141: "wheelchair",
                142: "wind turbine",
                143: "windmill",
                144: "window",
                145: "wristwatch"
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
                "perceptions": perceptions,
                "objects": objects,
                "in_out_door": in_out_door
            }

            return informations

        def create_classes():
            
            if self.nr_classes == 5:
                classes_dict = {
                    "Negative": 0,
                    "SlightlyNegative": 1,
                    "Neutral": 2,
                    "SlightlyPositive": 3,
                    "Positive": 4
                }
                classes_list = ["Negative", "SlightlyNegative", "Neutral", "SlightlyPositive", "Positive"]
            elif self.nr_classes == 2:
                if self.shift_neutral=="positive": 
                    if self.expand_neutral:
                        classes_dict = {
                            "Negative": 0,
                            "SlightlyNegative": 1,
                            "Neutral": 1,
                            "SlightlyPositive": 1,
                            "Positive": 1
                        }
                    else:
                        classes_dict = {
                            "Negative": 0,
                            "SlightlyNegative": 0,
                            "Neutral": 1,
                            "SlightlyPositive": 1,
                            "Positive": 1
                        }
                
                    classes_list = ["Negative", "Other"]
                elif self.shift_neutral=="negative": 
                    if self.expand_neutral:
                        classes_dict = {
                            "Negative": 0,
                            "SlightlyNegative": 0,
                            "Neutral": 0,
                            "SlightlyPositive": 0,
                            "Positive": 1
                        } 
                    else:
                        classes_dict = {
                            "Negative": 0,
                            "SlightlyNegative": 0,
                            "Neutral": 0,
                            "SlightlyPositive": 1,
                            "Positive": 1
                        }   
                
                    classes_list = ["Other", "Positive"]
                else:
                    raise Exception("Error while processing the shift neutral parameter.")
            elif self.nr_classes == 3:
                if self.expand_neutral:
                    classes_dict = {
                        "Negative": 0,
                        "SlightlyNegative": 1,
                        "Neutral": 1,
                        "SlightlyPositive": 1,
                        "Positive": 2
                    }
                else:
                    classes_dict = {
                        "Negative": 0,
                        "SlightlyNegative": 0,
                        "Neutral": 1,
                        "SlightlyPositive": 2,
                        "Positive": 2
                    }
                classes_list = ["Negative", "Neutral", "Positive"]
            else:
                raise Exception("Erro while processing the classes dictionary.")
            return classes_dict, classes_list
    
        num_seq = str(cfg[0])
        model = cfg[1]
        nr_classes = int(cfg[2])
        expand_neutral = (cfg[3].lower()=="true")
        shift_neutral = str(cfg[4])
        method = str(cfg[5])
        data_augmented = (cfg[6].lower()=="true")

        if cfg[7].lower() == "aggregated":
            metadata = "aggregated"
        elif cfg[7].lower() == "individual":
            metadata = "individual"
        else:
            metadata = False

        if cfg[8].lower() == "aggregated":
            perceptions = "aggregated"
        elif cfg[8].lower() == "individual":
            perceptions = "individual"
        else:
            perceptions = False

        object_detection = (cfg[9]=="true")

        if "dominant" in method and (metadata == "individual" or perceptions == "individual"):
            raise Exception("It's not possible to process aggregated sentiment with individual metadata or/and perceptions.")

        if "individual" in method and metadata == "aggregated":
            raise Exception("It's not possible to process individual sentiment with aggregated metadata.")

        strategy = str(cfg[10])

        if cfg[11]: 
            size = int(cfg[11])
        else:
            size = None

        if cfg[12]:
            class_balance = float(cfg[12].replace(",","."))
        else:
            class_balance = None

        self.model = model
        self.num_seq = num_seq
        self.method = method
        self.perceptions = perceptions
        self.object_detection = object_detection
        self.metadata = metadata
        self.nr_classes = nr_classes
        self.expand_neutral = expand_neutral
        self.shift_neutral = shift_neutral
        classes_dict, classes_list = create_classes()
        self.classes_dict = classes_dict
        self.classes_list = classes_list
        self.codes= create_coded_informations()
        self.labels = create_label_informations()
        self.data_augmented = data_augmented
        self.class_balance = class_balance
        self.strategy = strategy
        self.size = size
        self.profiling = profiling

        attributes = list()
        if metadata:
            attributes += ["age", "gs", "eco", "edu", "opt1", "neg1", "opt2"]
        if perceptions:
            attributes += ["perceptions"]
        if object_detection:
            attributes += ["objects"]
        self.attributes = attributes

        self.data = None
        self.imgs_list = list()
        self.imgs_agg = list()
        self.img_df = None
        
        self.create_dir(f"{output}")
        self.output_folder = self.create_dir(f"{output}/{self.num_seq}")
        self.dataset_json = f"{dataset}/dataset.json"
        self.dataset_folder = self.create_dir(f"{output}/{self.num_seq}/dataset")
        self.dataset_object_detection = f"{dataset}/object_detection"
        if self.attributes:
            self.create_dir(f"{output}/{self.num_seq}/dataset/descriptors")

        # Check the folder to retrieve the images.
        if self.method == "individual":
            self.images = f"{dataset}/images/individual/original"
        else:
            self.images = f"{dataset}/images/dominant/original"
        
        # Check if data is augmented.
        if self.data_augmented:
            self.images_data_augmented = f"{dataset}/images/individual/data_augmented"
        else:
            self.images_data_augmented = None

    def create(self):
  
        self.__extract_data()
        self.__aggregate()
        self.__calibrate()
        self.__create_dataset()
        self.__report()
        self.__save_list()
        self.__save_descriptors()
        self.__export_data()  

    def create_dir(self, dir):
        if not os.path.isdir(dir):
                try:
                    os.mkdir(dir)
                    return dir
                except Exception as e:
                    print(f"Creation of directory {dir} failed: {e}")
                    exit(1)
        else:
            return dir

    def __calculate_class_balance(self):

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

        balance_count = get_classes_count(self.nr_classes)
        for img in self.imgs_list:
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
            balance_imgs[cls_label] = [i for i in self.data if i[2]==int(cls_label)]
            balance_perc[cls_label] = balance_count[cls_label]/balance_count[cls_min]
        return balance_count, balance_perc, balance_imgs

    def __to_dict(self):
        cfg = dict()
        cfg["method"] = self.method
        cfg["cnn"] = self.model
        cfg["nr_classes"] = self.nr_classes
        cfg["expand_neutral"] = self.expand_neutral
        cfg["shift_neutral"] = self.shift_neutral
        cfg["data_augmented"] = self.data_augmented
        cfg["metadata"] = self.metadata
        cfg["perceptions"] = self.perceptions
        cfg["objet_detection"] = self.object_detection
        cfg["descriptors"] = self.attributes
        cfg["class_balance"] = self.class_balance
        cfg["strategy"] = self.strategy
        cfg["size"] = self.size
        return cfg

    def __extract_data(self):

        def read_data(file_in):
            with open(file_in, 'r') as j:
                json_data = json.load(j)
            tasks = json_data.get("tasks")
            return tasks

        def get_workers_info(tasks):
            workers_info = list()
            workers_list = list()
            keys=["worker_id", "worker_info"]
            for assgn in tasks:
                worker_id = assgn.get('worker_id')
                if not worker_id in workers_list:
                    worker_info = [x for x in tasks if x["worker_id"] == worker_id and x["worker_info"]]
                    if worker_info:
                        workers_info.append(dict(zip(keys, list(map(worker_info[0].get, keys)))))
                        workers_list.append(worker_id)
            return workers_info

        def get_info_resp(assgn, image, workers_info):
            worker_id = assgn.get('worker_id')
            worker_info = [x for x in workers_info if x["worker_id"] == worker_id and x["worker_info"]]
            if worker_info:
                data = worker_info[0].get("worker_info")
                data["perceptions"] = image.get("perceptions", [])
                return data
            else:
                return {
                        "age": "no_age",
                        "gs": "no_gender",
                        "eco": "no_eco",
                        "edu": "no_edu",
                        "opt1": "no_opt1",
                        "neg1": "no_neg1",
                        "opt2": "no_opt2",
                        "perceptions": image.get("perceptions", [])
                        }

        tasks = read_data(self.dataset_json)
        workers_info = get_workers_info(tasks)
        imgs = list()

        informations = self.codes
        desc_template = list()
        n_atts = 0
        for i, desc_label in enumerate(self.attributes):
            desc = informations.get(desc_label)
            class_max = max(desc, key=desc.get)
            tam = desc.get(class_max)
            desc_list = [0] * (tam+1)
            n_atts += len(desc_list)
            desc_template.append(desc_list)
        self.n_atts = n_atts

        for assgn in tasks:
            image_resps = assgn.get('images')  
            for image in image_resps:
                record = list()
                record.append(image.get('id'))
                record.append(assgn.get('assignment_id'))
                record.append(self.classes_dict.get(image.get('sentiment')))

                if self.attributes:
                    info_resp = get_info_resp(assgn, image, workers_info)
                
                att = dict()  
                for desc_label in self.attributes:
                    cat = None
                    cat_id = None
                    desc = informations.get(desc_label)
                    desc_list = [0] * (desc.get(max(desc, key=desc.get))+1)
                    
                    if desc_label in ["age", "gs", "eco", "edu", "opt1", "neg1", "opt2"]:
                        cat = info_resp.get(desc_label)
                        cat_id = informations.get(desc_label).get(cat)
                        desc_list[cat_id]+=1
                        if att.get("metadata", None):
                            att["metadata"][desc_label] = desc_list
                        else:
                            att["metadata"] = dict()
                            att["metadata"][desc_label] = desc_list
                    elif desc_label == "perceptions":
                        for cat in image.get(desc_label, []):
                            cat_id = informations[desc_label].get(cat, -1)
                            if cat_id >0:
                                desc_list[cat_id]+=1
                        att["perceptions"] = desc_list
                    elif desc_label == "objects":
                        try: 
                            tree = ET.parse(f"{self.dataset_object_detection}/{image.get('id')}.xml")
                            root = tree.getroot()
                            objects = root.findall("./object/name")
                        except:
                            objects = list()
                        for obj in objects:
                            cat = obj.text
                            cat_id = informations[desc_label].get(cat, -1)
                            if cat_id >0:
                                desc_list[cat_id]=1
                        att["objects"] = desc_list
                    else:
                        raise Exception("Unknown attribute!")
                
                record.append(att)
                imgs.append(record)    

        self.data = imgs

    def __aggregate(self):

        def dominant(imgs, nr_classes, threshold):
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

        if self.method == "dominant3":
            imgs_agg = dominant(self.data, self.nr_classes, 3)
        elif self.method == "dominant4":
            imgs_agg = dominant(self.data, self.nr_classes, 4)
        elif self.method == "dominant5":
            imgs_agg = dominant(self.data, self.nr_classes, 5)
        elif self.method == "individual":
            imgs_agg = self.data
        else:
            raise Exception("Method not implemented.")
        imgs_summ_np = np.asarray(imgs_agg)
        imgs_summ_name = imgs_summ_np[:,0]
        imgs_list = [img for img in self.data if img[0] in imgs_summ_name]
        self.imgs_list = imgs_list
        self.imgs_agg = imgs_agg

    def __calibrate(self):

        if not self.class_balance:
            return None

        assert (self.method!="individual"), "The method individual doesn't allow to apply class balance calibration."
        imgs_agg_bal = list()
        balance_count, balance_perc, balance_imgs = self.calculate_class_balance()    
        for cls_label, cls_perc in balance_perc.items():
            if cls_perc > self.class_balance:
                red_coef = self.class_balance/cls_perc
                reduction = balance_count.get(cls_label) - (balance_count.get(cls_label) * red_coef)
                imgs_cls = balance_imgs.get(cls_label)
                while reduction > 0:
                    shuffle(imgs_cls)
                    item = imgs_cls.pop(0)
                    reduction -= 1
            imgs_agg_bal += balance_imgs[cls_label]  
        imgs_list_bal = list()
        for img in imgs_agg_bal:
            imgs_list_bal += [i for i in self.imgs_list if i[0]==img[0]]
        self.imgs_list = imgs_list_bal
        self.imgs_agg = imgs_agg_bal

    def __create_dataset(self):
        
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
                    img_votes.append([0] * self.nr_classes)
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
            for c in range(self.nr_classes):
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
            for c in range(self.nr_classes):
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
            for c in range(self.nr_classes):
                wkr_stat_group_df.drop([f"sent_{c}"], axis='columns', inplace=True)
            wkr_stat_group_df = wkr_stat_group_df.sort_values(by=["wkr_agr", "wkr_dis", "wkr"], ascending=[True, False, True])
            del wkr_stat_df

            return img_stat_df, wkr_stat_group_df

        def img_agr_dataset(img_stat_df):
            # Creating the datasets - Image quality
            df = img_stat_df.sort_values(by=["img_agr", "img_dis", "img"], ascending=[False, True, True])
            img_agr = list()
            for c in range(self.nr_classes):
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
            for c in range(self.nr_classes):
                columns.append(f"sent_{c}")
            columns += ["sent_sum_quad", "jdg_1", "jdg_2", "jdg_3", "jdg_4", "jdg_5", "jdg_sum", "sent", "avg_img_wkr_agr", "avg_img_wkr_dis", "wkrs"]
            df = pd.DataFrame(imgs_full, columns=columns)
            wkr_agr = list()
            for c in range(self.nr_classes):
                wkr_agr.append(df.loc[df["sent"] == c])
            del imgs, w_wkrs, filter_wkrs, w_img, w, columns, df, c

            return wkr_agr

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
                metrics.append(self.__build_metrics(df, False))
            columns = ["img_agr", "img_dis", "avg_img_wkr_agr", "avg_img_wkr_dis", "fless", "cronbach", "imgs"]
            for c in range(nr_classes):
                columns.append(f"imgs_{c}")
            metrics_df = pd.DataFrame(metrics, columns=columns)
            metrics_df.to_csv(f"{file_name}.csv")

        if self.size == None:
            if self.method=="individual":
                self.size = len(self.imgs_list) 
            else: 
                self.size = len(self.imgs_agg) 

        img_stat_df, wkr_stat_df = process_dataset(self.data, self.imgs_agg)

            
        if self.strategy == "wkr_agr":
            df_list = wkr_agr_dataset(img_stat_df, wkr_stat_df)
        else:
            df_list = img_agr_dataset(img_stat_df)
        
        if self.profiling:
            build_metrics_profile(df_list, self.nr_classes, f"{self.dataset_folder}/profiling.csv")
            
        df_list_head = list()
        for c in range(self.nr_classes):
            df_list_head.append(df_list[c].head(self.size))
        img_df = pd.concat(df_list_head)
        img_index = img_df["img"].values.tolist()

        self.imgs_agg = [img for img in self.imgs_agg if img[0] in img_index]
        self.imgs_list = [img for img in self.imgs_list if img[0] in img_index]
        self.img_df = img_df

    def __build_metrics(self, df=None, dict_fmt=True):

        def fleiss(df):
            # Calculate Fleiss.
            m = 5
            n = len(df.index)
            q = list()
            pe = 0.0
            for c in range(self.nr_classes):
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

        if not df:
            df = self.img_df

        metrics = list()
        metrics.append(df["img_agr"].mean())
        metrics.append(df["img_dis"].mean())
        metrics.append(df["avg_img_wkr_agr"].mean())
        metrics.append(df["avg_img_wkr_dis"].mean())
        metrics.append(fleiss(df))
        metrics.append(cronbach(df))
        metrics.append(len(df.index))
        for c in range(self.nr_classes):
            metrics.append(len(df.loc[df["sent"] == c].index))    
        
        if dict_fmt:
            columns = ["img_agr", "img_dis", "avg_img_wkr_agr", "avg_img_wkr_dis", "fless", "cronbach", "imgs"]
            for c in range(self.nr_classes):
                columns.append(f"imgs_{c}")
            metrics_df = pd.DataFrame([metrics], columns=columns)
            return metrics_df.to_dict()
        else:
            return metrics

    def __report(self):
        balance_count, balance_perc, _ = self.__calculate_class_balance()  
        metrics = self.__build_metrics()
        with open(f"{self.dataset_folder}/metadata.txt", 'w') as f:
            f.write(f"{json.dumps(self.__to_dict())}")
            f.write("\n")
            f.write(f"{','.join(self.classes_list)}")
            f.write("\n")
            f.write(f"{json.dumps(self.classes_dict)}")
            f.write("\n")
            f.write(f"{json.dumps(balance_count)}")
            f.write("\n")
            f.write(f"{json.dumps(balance_perc)}")
            f.write("\n")
            f.write(f"{json.dumps(metrics)}")
        with open(f"{self.dataset_folder}/report.txt", 'w') as f:
            f.write(f"Configuration: {json.dumps(self.__to_dict(), indent=2, default=str)}")
            f.write("\n\n")
            f.write(f"Classes (out): {json.dumps(self.classes_list, indent=2, default=str)}")
            f.write("\n\n")
            f.write(f"Classes (in): {json.dumps(self.classes_dict, indent=2, default=str)}")
            f.write("\n\n")
            f.write(f"Metrics: {json.dumps(metrics, indent=2, default=str)}")
            f.write("\n\n")
            f.write(f"Balance count: {json.dumps(balance_count, indent=2, default=str)}")
            f.write("\n\n")
            f.write(f"Balance %: {json.dumps(balance_perc, indent=2, default=str)}")

    def __save_list(self):

        if self.data_augmented and self.method!="individual":

            img_summ_i = [i[0] for i in self.imgs_agg]
            img_summ_2 = list()
            for i in self.img_list:
                if i[0] in img_summ_i:
                    index = img_summ_i.index(i[0])
                    img_summ_2.append([i[0], i[1], self.imgs_agg[index][2]])
            self.imgs_agg = img_summ_2

        if self.method == "individual" or self.data_augmented:
            names = list()
            workers = list()
            for img in self.imgs_agg:
                if img[0] not in names:
                    names.append(img[0])
                    workers.append(f"{img[1]} {str(img[2])}")
                else:
                    index = names.index(img[0])
                    workers[index] += f" {img[1]} {str(img[2])}"

            with open(f"{self.dataset_folder}/data.txt", 'w') as f:
                for img in zip(names, workers):
                    f.write(f"{img[0]} {img[1]}\n")
        else:
            with open(f"{self.dataset_folder}/data.txt", 'w') as f:
                for img in self.imgs_agg:
                    f.write(f"{img[0]} {str(img[2])}\n")
        
        self.dataset_file = f"{self.dataset_folder}/data.txt"

    def __save_descriptors(self):

        if not self.attributes:
            self.attributes = None
            return None

        imgs = list()
        workers = list()
        metadata = list()
        perceptions = list()
        perceptions_agg = list()
        objects = list()
        imgs_agg_temp = list()
        perc_agg_temp = list()             
        
        for img in self.data:  
            img_name = img[0]
            img_p = img[3].get("perceptions", [])
            if img_name not in imgs_agg_temp:
                imgs_agg_temp.append(img[0])
                perc_agg_temp.append(img_p)
            else:
                img_id = imgs_agg_temp.index(img_name)   
                perc_agg_temp[img_id] = [x + y for x, y in zip(perc_agg_temp[img_id], img_p)]

        for img in self.data:

            if self.method == "individual":
                img_name = f"{img[0]}_{img[1]}"
            else:
                img_name = img[0]
            
            atts = img[3]
            img_m = list()
            img_m_dict = atts.get("metadata", dict())
            for m in img_m_dict.values():
                img_m += m
            img_p = atts.get("perceptions", [])
            img_o = atts.get("objects", [])
            img_pa = perc_agg_temp[imgs_agg_temp.index(img[0])]

            if img_name not in imgs:
                imgs.append(img_name)
                workers.append(list())
                metadata.append(img_m)
                perceptions.append(img_p)
                perceptions_agg.append(img_pa)
                objects.append(img_o)
            else:
                img_id = imgs.index(img_name)   
                workers[img_id].append(img[1])
                metadata[img_id] = [x + y for x, y in zip(metadata[img_id], img_m)]
                perceptions[img_id] = [x + y for x, y in zip(perceptions[img_id], img_p)]
                perceptions_agg.append(img_pa)
                objects[img_id] = img_o        

        descriptors = list()
        if self.method == "individual" and self.perceptions == "aggregated":
            descriptors = [m+p+o for m, p, o in zip(metadata, perceptions_agg, objects)]
        else:
            descriptors = [m+p+o for m, p, o in zip(metadata, perceptions, objects)]
            
        file_list = list()
        desc_content = list()
        if self.data_augmented:
            for i, img in enumerate(imgs):
                for wkr in workers[i]:
                    file_list.append(f"{img}_{wkr}.txt")
                    desc_content.append(descriptors[i])
        else:
            for i, img in enumerate(imgs):
                file_list.append(f"{img}.txt")
                desc_content.append(descriptors[i])

        for file_name, content in zip(file_list, desc_content):
            np.savetxt(f"{self.dataset_folder}/descriptors/{file_name}", np.asarray(content, dtype="float32")) 
        self.attributes = f"{self.dataset_folder}/descriptors"

    def __export_data(self):

        def get_value(dict_values, list_choice):
            for index, c in enumerate(list_choice):
                if c == 1:
                    return dict_values.get(index)

        headers = ["image", "worker", "sentiment", "age", "gs", "eco", "edu", "opt1", "neg1", "opt2", "perceptions", "objects"]
        data_lb = list()
        data_cd = list()
        data_lb.append(headers)
        data_cd.append(headers)
        
        for img in self.data:

            img_id = img[0]
            wkr_id = img[1]
            sent_cd = img[2]
            sent_lb = self.classes_list[sent_cd]  
            perceptions_cd = list()
            perceptions_lb = list()
            p_labels = self.labels.get("perceptions", dict())
            for index, p in enumerate(img[3].get("perceptions", [])):
                if p:
                    perceptions_cd.append(index)
                    perceptions_lb.append(p_labels.get(index, ""))

            objects_cd = list()
            objects_lb = list()
            o_labels = self.labels.get("objects", dict())
            for index, o in enumerate(img[3].get("objects", [])):
                for _ in range(o):
                    objects_cd.append(index)
                    objects_lb.append(o_labels.get(index, ""))
                    
            metadata = img[3].get("metadata", None)
            if metadata:
                age_lb = get_value(self.labels["age"], metadata["age"])
                gs_lb = get_value(self.labels["gs"], metadata["gs"])
                eco_lb = get_value(self.labels["eco"], metadata["eco"])
                edu_lb = get_value(self.labels["edu"], metadata["edu"])
                opt1_lb = get_value(self.labels["opt1"], metadata["opt1"])
                neg1_lb = get_value(self.labels["neg1"], metadata["neg1"])
                opt2_lb = get_value(self.labels["opt2"], metadata["opt2"])

                age_cd = self.codes["age"].get(age_lb)
                gs_cd = self.codes["gs"].get(gs_lb)
                eco_cd = self.codes["eco"].get(eco_lb)
                edu_cd = self.codes["edu"].get(edu_lb)
                opt1_cd = self.codes["opt1"].get(opt1_lb)
                neg1_cd = self.codes["neg1"].get(neg1_lb)
                opt2_cd = self.codes["opt2"].get(opt2_lb)

            else:
                age_lb = None
                gs_lb = None
                eco_lb = None
                edu_lb = None
                opt1_lb = None
                neg1_lb = None
                opt2_lb = None

                age_cd = None
                gs_cd = None
                eco_cd = None
                edu_cd = None
                opt1_cd = None
                neg1_cd = None
                opt2_cd = None


            data_cd.append([img_id, wkr_id, sent_cd, 
                                age_cd, gs_cd, eco_cd, edu_cd, opt1_cd, neg1_cd, opt2_cd,
                                ",".join([str(i) for i in perceptions_cd]),
                                ",".join([str(i) for i in objects_cd])])

            data_lb.append([img_id, wkr_id, sent_lb,
                                age_lb, gs_lb, eco_lb, edu_lb, opt1_lb, neg1_lb, opt2_lb,
                                ",".join([str(i) for i in perceptions_lb]),
                                ",".join([str(i) for i in objects_lb])])
        
        data_lb_t = list()
        data_lb_t.append(["image", "worker", "data", "field"])
        for dd in data_lb[1:]:
            for i, d in enumerate(dd[2:10]):
                if d is not None: data_lb_t.append([dd[0], dd[1], d, data_lb[0][i+2]])
            for r in dd[10].split(','):
                if r!='': data_lb_t.append([dd[0], dd[1], r, "perceptions"])
            for r in dd[11].split(','):
                if r!='': data_lb_t.append([dd[0], dd[1], r, "objects"])
        
        data_cd_t = list()
        data_cd_t.append(["image", "worker", "data", "field"])
        for dd in data_cd[1:]:
            for i, d in enumerate(dd[2:10]):
                if d is not None: data_cd_t.append([dd[0],dd[1], d, data_cd[0][i+2]])
            for r in dd[10].split(','):
                if r!='': data_cd_t.append([dd[0], dd[1], r, "perceptions"])
            for r in dd[11].split(','):
                if r!='': data_cd_t.append([dd[0], dd[1], r, "objects"])
        
        with open(f'{self.dataset_folder}/raw_label_t.csv', 'w') as f:
            write = csv.writer(f, delimiter=';')
            for i in data_lb_t:
                write.writerow(i)
        
        with open(f'{self.dataset_folder}/raw_code_t.csv', 'w') as f:
            write = csv.writer(f, delimiter=';')
            for i in data_cd_t:
                write.writerow(i)


        with open(f'{self.dataset_folder}/raw_label.csv', 'w') as f:
            write = csv.writer(f, delimiter=';')
            for i in data_lb:
                write.writerow(i)

        with open(f'{self.dataset_folder}/raw_code.csv', 'w') as f:
            write = csv.writer(f, delimiter=';')
            for i in data_cd:
                write.writerow(i)
