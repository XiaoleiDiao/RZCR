# -----------------------------------------------------------------------#
#    KGR
#    Reasoning characters in character knowledge graph
#    via radical predictions and structural relation predictions
# -----------------------------------------------------------------------#


import pandas
import os
import cv2
import numpy as np
import rdflib
from rdflib import Namespace
import collections
import pandas as pd
from PIL import Image
import itertools
from get_RPs_SP import RIE
import time

def readKG(owl_path):
    '''
     Read Knowledge graph
    '''
    g1 = rdflib.Graph()
    g1.parse(owl_path, format="xml")
    ns = Namespace('http://www.jlu.edu.cn/CR/ontology#')  # 命名空间

    return(g1, ns)


def gerCharacterRadical(excelcharactername):
    '''
    Get character categories' name that corresponding with character labels
    '''
    xls_character = pd.read_excel(excelcharactername)
    character_name = []
    for x in xls_character.iloc[:, 1]:
        x = str(x)
        character_name.append(x)
    dic_character_name = collections.OrderedDict(zip(character_name, xls_character.iloc[:, 2]))
    dic_contain_radical = collections.OrderedDict(zip(character_name, xls_character.iloc[:, 5]))
    return dic_character_name, dic_contain_radical



def DerectionRadical(ImagePath):
    '''
    Predict radical information, output radical predictions and structural relation predictions.
    '''
    image = Image.open(ImagePath)
    model = RIE()
    RPs, SP, r_image = model.detect_image(image)
    return RPs, SP


def R_label_to_name(excel_path, labels):
    '''
    get radical name that corresponding with radical labels.
    '''
    xls = pd.read_excel(excel_path)
    dic = collections.OrderedDict(zip(xls.iloc[:, 0], xls.iloc[:, 1]))
    name_conf = []
    for i in range(len(labels)):
        name_conf_temp = []
        for j in range(len(labels[0])):
            temp = []
            label = 'r_'+ str(labels[i][j][0])
            name = dic.get(label)
            conf = labels[i][j][1]
            temp.append(name)
            temp.append(conf)
            temp_t = tuple(temp)
            name_conf_temp.append(temp_t)
        name_conf.append(name_conf_temp)
    return name_conf



def map(RPs):
    '''
    Get all candidate radical composition sets.
    '''
    R = []
    for i in range(len(RPs)):
        r_candidate = RPs[i]
        # maxSort the confidence of radical candidates
        r_candidate.sort(key=lambda x: x[1], reverse=True)
        R.append(r_candidate)
    if len(R) == 1:
        return R
    elif len(R) ==2:
        pairs = list(itertools.product(R[0], R[1]))
        return pairs
    elif len(R) ==3:
        pairs = list(itertools.product(R[0], R[1], R[2]))
        return pairs
    else:
        return R



def CharReason(CKG, RPs, SP, dic_contain_radical):
    '''
    Reasoning characters in Character knowledge graph with RPs and SP.
    '''
    character_can = []

    # mapping radicals
    M = map(RPs)

    # maxSort the confidence of structural relation candidates
    SP.sort(key=lambda x: x[1], reverse=True)

    # find all candidate radical composition sets
    for i in range(len(M)):
        character_temp = list(dic_contain_radical.keys())
        pr_all = 0
        for j in range(len(M[0])):
            r = M[i][j][0]

            # search characters in CKG with radicals, and get corresponding confidence
            character = []
            for s in CKG.subjects(ns['Contain'], ns[r]):
                c = s[34:]
                character.append(c)
            temp = character
            character_temp = list(set(character_temp).intersection(set(temp)))
            pr = M[i][j][1]
            pr_all = pr_all + pr

            # search characters in CKG with structural relations, and get corresponding confidence
            character_sr = []
            spj = []
            for k in range(len(SP)):
                sr = SP[k][0]
                for s in CKG.subjects(ns['Compose'], ns[sr]):
                    c = s[34:]
                    character_sr.append(c)
                spj.append(SP[k][1])

        # get candidate character confidence
        mi = pr_all / len(M[0])
        pc = 0.7*mi+0.3*spj[0]

        # get all candidate characters and their confidence
        for e in character_temp:
            character_conf = []
            if i == 0:
                character_conf.append(e)
                character_conf.append(pc)
                character_conf_t = tuple(character_conf)
                character_can.append(character_conf_t)
            else:
                if e in np.array(character_can)[:,0]:
                    text ="This character is exixting"
                else:
                    character_conf.append(e)
                    character_conf.append(pc)
                    character_conf_t = tuple(character_conf)

                    character_can.append(character_conf_t)

    character_can.sort(key=lambda x: x[1], reverse=True)
    # print("character_can", len(character_can), character_can)
    return character_can



if __name__ == '__main__':
    input_path = "F:\oracle\REZCR\img/oc_02_1_0139_1_3.png"
    owl_path= "./oracle_779_KG.owl"
    CKG, ns = readKG(owl_path)
    excelradicalname = './excel_data/59_radical_id.xls'
    excelcharactername = './excel_data/oracle_radical_779.xls'


    RPs_temp, SP = DerectionRadical(input_path)
    RPs = R_label_to_name(excelradicalname, RPs_temp)
    dic_character_name, dic_contain_radical = gerCharacterRadical(excelcharactername)

    # time_start = time.time()  # 记录开始时间
    time_start = time.perf_counter()
    print("start", time_start)
    # function()   执行的程序
    Character_candidate = CharReason(CKG, RPs, SP, dic_contain_radical)

    # time_end = time.time()  # 记录结束时间
    time_end = time.perf_counter()
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print("time_end", time_end)
    print("time_sum", time_sum)


    # Print TOP1 characters
    character = dic_character_name.get(Character_candidate[0][0])
    print("TOP1", character, Character_candidate[0][1])

    # Print TOP5 characters
    for i in range(len(Character_candidate[:5])):
        c = Character_candidate[i][0]
        p = Character_candidate[i][1]
        character = dic_character_name.get(c)
        print("TOP5-",i, character, p)

