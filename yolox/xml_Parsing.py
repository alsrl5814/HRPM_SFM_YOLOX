from __future__ import annotations
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree
from glob import glob
import os
xml_list = glob("F:/YOLOX/datasets/RDD2022/VOC2007/Annotations/*.xml")

for xml_name in xml_list:
    print(xml_name) 
    tree = ET.parse(xml_name)
    root = tree.getroot()
    #objectness = annotation.find("object")
    objectness = root.findall("object")
    
    #print(objectness[1].find("name").text)
    #print(len(objectness))


    for objectness1 in root.iter("object"):
        print(len(objectness1))
        print(objectness1.find("name").text)
        if objectness1.find("name").text == "D40":
            objectness1.find("name").text = "pothole"
        elif objectness1.find("name").text == "D00" or objectness1.find("name").text == "D10" or objectness1.find("name").text == "D20":
            objectness1.find("name").text = "crack"
            
        else:
            objectness1.find("name").text = "no"
            
        

    tree = ElementTree(root)

    tree.write(xml_name)