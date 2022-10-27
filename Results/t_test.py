import csv
import json
from scipy import stats

file = "Results/fscore.csv"
alpha = 0.05

exp_dict = dict()
with open(file) as csv_file:
    heading = next(csv_file)
    reader = csv.reader(csv_file, delimiter=';')
    for row in reader:
        values = exp_dict.get(row[0], list())
        values.append(float(row[1].replace(",", ".")))
        exp_dict[row[0]] = values

res_dict = dict()
for key1, value1 in exp_dict.items():
    for key2, value2 in exp_dict.items():
        results = res_dict.get(key1, dict())
        key_result = results.get(key2, None)
        if key_result is not None:
            continue 
        p_value = stats.ttest_ind(value1, value2, equal_var=False).pvalue
        if p_value<alpha:
            t_test = "Different"
        else:
            t_test = "Equal"
        if key1 == key2:
            t_test = "-"
        results2 = res_dict.get(key2, dict())
        key_result2 = results2.get(key1, None)
        if key_result2 is not None:
            t_test = "-" 
        results[key2] = t_test
        res_dict[key1] = results

with open("Results/t_test.json", "w") as outfile:
    json.dump(res_dict, outfile)