import os
import numpy as np
import pandas as pd

def get_rules(movi_list):
    data = pd.read_pickle(os.getcwd()+"/models/association_rules/rules.pkl")
    retvalue = []
    while len(movi_list) > 1 and len(retvalue) <= 0:
        retvalue = []
        for index, row in data.iterrows():
            if row['antecedents'] == movi_list:
                retvalue.extend(row["consequents"])
        if(len(retvalue) > 0):
            break
        else:
            del movi_list[0]
    retvalue = np.unique(retvalue)
    return list(retvalue)