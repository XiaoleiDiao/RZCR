from xml.dom import minidom
import os
import cv2
import pandas as pd


def readFiles(tpath):
    txtLists = os.listdir(tpath)
    List = []
    for t in txtLists:
        t = tpath + "/" + t
        List.append(t)
    return List


def XML_write(xmlExample, name_O, o):
    dom = minidom.parse(xmlExample)
    root = dom.documentElement
    names = root.getElementsByTagName('filename')
    names[0].childNodes[0].nodeValue = name_O + '.jpg'

    # name = names[0].childNodes[0].nodeValue.split(".")[0]
    # width = root.getElementsByTagName('width')[0].childNodes[0].nodeValue
    # height = root.getElementsByTagName('height')[0].childNodes[0].nodeValue
    # label = root.getElementsByTagName('name')[0].childNodes[0].nodeValue
    # x_min = root.getElementsByTagName('xmin')[0].childNodes[0].nodeValue
    # y_min = root.getElementsByTagName('ymin')[0].childNodes[0].nodeValue
    # x_max = root.getElementsByTagName('xmax')[0].childNodes[0].nodeValue
    # y_max = root.getElementsByTagName('ymax')[0].childNodes[0].nodeValue
    # print(width,height)
    # print(name)
    # print(label)
    # print(x_min,y_min,x_max,y_max)

    path_oracle = 'G:\BACKUP\learning\programs\OracleRecognition\OracleDetection\DataPreparing-pytorch\yolo3-pytorch-radical\VOCdevkit\VOC2007\JPEGImages/'
    pp = path_oracle + name_O + '.jpg'
    root.getElementsByTagName('path')[0].childNodes[0].nodeValue = pp

    img = cv2.imread(o, 0)
    w, h = img.shape[::-1]
    root.getElementsByTagName('width')[0].childNodes[0].nodeValue = w
    root.getElementsByTagName('height')[0].childNodes[0].nodeValue = h

    root.getElementsByTagName('name')[0].childNodes[0].nodeValue = name_O.split('_')[1]
    root.getElementsByTagName('xmin')[0].childNodes[0].nodeValue = OB_results.loc[name_O,'X0']
    root.getElementsByTagName('ymin')[0].childNodes[0].nodeValue = OB_results.loc[name_O,'Y0']
    root.getElementsByTagName('xmax')[0].childNodes[0].nodeValue = OB_results.loc[name_O,'X1']
    root.getElementsByTagName('ymax')[0].childNodes[0].nodeValue = OB_results.loc[name_O,'Y1']


    XML_writer(dom, name_O, xml_store_path)



def XML_writer(dom, name, path_store='../Datasets/OracleRC2022/Annotations/'):
    name = path_store + name + '.xml'
    with open(name, 'w', encoding='utf-8') as f:
        dom.writexml(f, addindent='\t', newl='\n', encoding='utf-8')


img_path = '../Datasets/OracleRC2022/JPEGImages'
xmlExample = 'exampleXML.xml'
xml_store_path = '../Datasets/OracleRC2022/Annotations/'
OB_results = pd.read_csv('rect_Oracles.csv', index_col='Name')

oracles = readFiles(img_path)
for o in oracles:
    # Get radical label
    name = o.split('/')[-1].split('.')[0]
    print(name)
    if name in OB_results.index.tolist():
        try:
            output = XML_write(xmlExample, name, o)

        except Exception as e:
            print('ERROR:' + name)
            print(e)
