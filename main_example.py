# !/usr/bin/env python
# encoding: utf-8
"""         
  @Author: Yongheng Sun          
  @Contact: 3304925266@qq.com          
  @Software: PyCharm    
  @Project: transformer_0322
  @File: main_example.py          
  @Time: 2022/3/12 15:47                   
"""
# from Models.networks.network import Comprehensive_Atten_Unet

# from Models.networks.csnet_family import CSNet_no_ca, CSNet_no_sa, CSNet_no_casa, CSNet, CSNet_rfb, CSNet_rfb7a, CSNet_rfb7b
# from Models.networks.csnet_family import CSNet_hs, CSNet_hsrfb, CSNet_hsrfb_att, CSNet_hsrfb_att_scale, CSNet_hsrfb_att_assf
# from Models.networks.csnet_family import CSNet_hsrfb_att_assf_scale, CSNet_hsrfb_att_assf_scale_sfpool
from Models.networks.csnet_rfb_final import CSNet_rfb_hs_assf_scale_sfpool_all, CSNet_hsrfb7a_assf_scale_sfpool_all
from Models.networks.csnet_rfb_final import CSNet_rfb_hs_scale_scale_sfpool_all
from Models.compare_networks.BCDU_Net import BCDU_net_D3
from Models.compare_networks.Focusnet_Alpha import Focus_alpha_net
from Models.compare_networks.CPFNet import CPF_Net
from Models.compare_networks.CENet import CE_Net
from Models.compare_networks.CENet_v1 import CE_Net_v1
from Models.compare_networks.CENet_v2 import CE_Net_v2
from Models.compare_networks.CENet_v3 import CE_Net_v3
from Models.compare_networks.CENet_v4 import CE_Net_v4
from Models.compare_networks.CENet_v11 import CE_Net_v11
from Models.compare_networks.DONet import DO_Net
from Models.compare_networks.biDFL_AND_mCDF import biDFL_AND_mCDF_net
from Models.compare_networks.DDW_CENet import DDW_CE_Net
from Models.compare_networks.DDW_CENet_resnest import DDW_CE_Net_resnest

# from Models.DeepLabV3Plus.network import deeplabv3plus_resnet50
# from Models.UnetSeries.network import U_Net, AttU_Net
# from Models.UnetSeries.network import NestedUNet as U_Net_plusplus
# from Models.denseaspp.models.DenseASPP_ddw import DenseASPP_161, DenseASPP_121
# from Models.FCN.fcn import FCN32s_vgg16, FCN16s_vgg16, FCN8s_vgg16
from thop import profile

from Models.networks.network import Comprehensive_Atten_Unet

from Models.networks.csnet_family import CSNet_no_ca, CSNet_no_sa, CSNet_no_casa, CSNet, CSNet_rfb, CSNet_rfb7a, CSNet_rfb7b
from Models.networks.csnet_family import CSNet_hs, CSNet_hsrfb, CSNet_hsrfb_att, CSNet_hsrfb_att_scale, CSNet_hsrfb_att_assf
from Models.networks.csnet_family import CSNet_hsrfb_att_assf_scale, CSNet_hsrfb_att_assf_scale_sfpool
from Models.networks.csnet_rfb_final import CSNet_rfb_hs_assf_scale_sfpool_all, CSNet_hsrfb7a_assf_scale_sfpool_all
from Models.networks.csnet_rfb_final import CSNet_rfb_hs_scale_scale_sfpool_all

from Models.DeepLabV3Plus.network import deeplabv3plus_resnet50
from Models.UnetSeries.network import U_Net, AttU_Net
from Models.UnetSeries.network import NestedUNet as U_Net_plusplus
from Models.denseaspp.models.DenseASPP_ddw import DenseASPP_161, DenseASPP_121
from Models.FCN.fcn import FCN32s_vgg16, FCN16s_vgg16, FCN8s_vgg16

import torch
if __name__ == '__main__':
    img = torch.rand(2,3,224,320)
    model = DenseASPP_121()
    out = model(img)
    flops, params = profile(model, inputs=(torch.rand(1, 3, 224,320), ))
    print('Params = ' + str(params/1000**2) + 'M')
    print('FLOPs = ' + str(flops/1000**3) + 'G')