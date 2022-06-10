# !/usr/bin/env python
# encoding: utf-8
"""         
  @Author: Yongheng Sun          
  @Contact: 3304925266@qq.com          
  @Software: PyCharm    
  @Project: transformer_0322
  @File: make_json.py          
  @Time: 2022/3/4 9:38                   
"""
import glob
import os
import re
import json
from collections import OrderedDict


#将YOUR DIR替换成你自己的目录
path_originalData = "./Task555_ISIC2018/"

# os.mkdir(path_originalData+"imagesTr/")
# os.mkdir(path_originalData+"labelsTr/")
# os.mkdir(path_originalData+"imagesTs/")
# os.mkdir(path_originalData+"labelsTs/")



def list_sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]
    l.sort(key=alphanum_key)
    return l



train_image = list_sort_nicely(glob.glob(path_originalData+"imagesTr/*"))
train_label = list_sort_nicely(glob.glob(path_originalData+"labelsTr/*"))
test_image = list_sort_nicely(glob.glob(path_originalData+"imagesTs/*"))
test_label = list_sort_nicely(glob.glob(path_originalData+"labelsTs/*"))

train_image = ["{}".format(patient_no.split('/')[-1]) for patient_no in train_image]
train_label = ["{}".format(patient_no.split('/')[-1]) for patient_no in train_label]
test_image = ["{}".format(patient_no.split('/')[-1]) for patient_no in test_image]
#输出一下目录的情况，看是否成功
print(len(train_image),len(train_label),len(test_image),len(test_label), train_image[0])



#####下面是创建json文件的内容
#可以根据你的数据集，修改里面的描述
json_dict = OrderedDict()
json_dict['name'] = "Task555_ISIC2018"
json_dict['description'] = " Segmentation"
json_dict['tensorImageSize'] = "4D"#???
json_dict['reference'] = "see challenge website"
json_dict['licence'] = 'hands off!'
json_dict['release'] = "0.0"
#这里填入模态信息，0表示只有一个模态，还可以加入“1”：“MRI”之类的描述，详情请参考官方源码给出的示例
json_dict['modality'] = {
    "0": "Red",
    "1": "Green",
    "2": "Blue"
}

#这里为label文件中的多个标签，比如这里有血管、胆管、结石、肿块四个标签，名字可以按需要命名
json_dict['labels'] = {
    "0": "Background",
    "1": "lesion ",#静脉血管
    # "2": "bileduck",#胆管
    # "3": "stone",#结石
    # "4": "lump" #肿块
}

#下面部分不需要修改>>>>>>
json_dict['numTraining'] = len(train_image) // 3
json_dict['numTest'] = len(test_image) // 3

json_dict['training'] = []
for idx in range(len(train_label)):
    json_dict['training'].append({'image': "./imagesTr/%s" % train_image[idx * 3][9:25] +'.nii.gz', "label": "./labelsTr/%s" % train_label[idx][9:25] +'.nii.gz'})
    # json_dict['training'].append({'image': "./imagesTr/%s" % train_image[idx * 3][9:25] +'.nii.gz', "label": "./labelsTr/%s" % train_label[idx][9:25] +'.nii.gz'})
    # json_dict['training'].append({'image': "./imagesTr/%s" % train_image[idx * 3][9:25] +'.nii.gz', "label": "./labelsTr/%s" % train_label[idx][9:25] +'.nii.gz'})
json_dict['test'] = []
for idx in range(len(test_image) // 3):
    json_dict['test'].append("./imagesTs/%s" % test_image[idx * 3][9:25] +'.nii.gz')
# json_dict['test'] = ["./imagesTs/%s" % i[9:25] +'.nii.gz' for i in test_image]

with open(os.path.join(path_originalData, "dataset.json"), 'w') as f:
    json.dump(json_dict, f, indent=4, sort_keys=True)
#<<<<<<<
