import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class NonparametricShift(object):
    def buildAutoencoder(self, target_img, normalize, interpolate,    nonmask_point_idx,          mask_point_idx, patch_size=1, stride=1):
                        #      [512,32,32],   False,   False,    nonmask 패치의 위치를 저장한 리스트, mask 위치 저장 리스트, 1, 1
        nDim = 3
        assert target_img.dim() == nDim, 'target image must be of dimension 3.' #512,32,32
        C = target_img.size(0)  #채널 수

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.Tensor

        patches_all, patches_part, patches_mask= self._extract_patches(target_img, patch_size, stride, nonmask_point_idx,mask_point_idx)
        # 모든 패치,   nonmask 패치, mask패치                         # [512,32,32],      1 ,       1  ,  인덱스 저장 리스트
        
        npatches_part = patches_part.size(0)#non-mask 패치 [x,512,1,1] 
        npatches_all = patches_all.size(0)  #모든 패치 1024개 [1024,512,1,1]  1024(32*32)  => 32*32피쳐가 512개 있는 상황.
        npatches_mask=patches_mask.size(0)  #mask 패치   [1024-x, 512, 1, 1]
        conv_enc_non_mask, conv_dec_non_mask = self._build(patch_size, stride, C, patches_part, npatches_part, normalize, interpolate)
        # 512 -> x, 32,32 |  x->512, 32, 32                   # 1         1   512 ex)493,512,1,1    493            F          F
        conv_enc_all, conv_dec_all = self._build(patch_size, stride, C, patches_all, npatches_all, normalize, interpolate)
        # 512 -> 1024 | 1024 -> 512                 1          1   512  [1024,512,1,1] 1024           F            F

        return conv_enc_all, conv_enc_non_mask, conv_dec_all, conv_dec_non_mask ,patches_part, patches_mask

    def _build(self, patch_size, stride, C, target_patches, npatches, normalize, interpolate):
                    #       1  ,    1 ,512, nonmask 패치,nonmask패치 개수, False, False
        
        # for each patch, divide by its L2 norm.
        enc_patches = target_patches.clone() #x,512,1,1 복사

        for i in range(npatches): # 패치 정규화 conv_enc의 가중치로 쓰기 위함.
            enc_patches[i] = enc_patches[i]*(1/(enc_patches[i].norm(2)+1e-8))

        conv_enc = nn.Conv2d(C, npatches, kernel_size=patch_size, stride=stride, bias=False)
        # in_channels : 512 
        # out_channels : x
        # 1*1 conv로 채널수 조정
        conv_enc.weight.data = enc_patches
        #512,32,32 feature를 x,32,32 feature로 변환한다. 그 때 사용한 가중치가 [x,512,1,1]을 정규화한 값
        
        # normalize is not needed, it doesn't change the result!
        if normalize:
            raise NotImplementedError

        if interpolate:
            raise NotImplementedError

        conv_dec = nn.ConvTranspose2d(npatches, C, kernel_size=patch_size, stride=stride, bias=False)
        conv_dec.weight.data = target_patches
        #decoder도 마찬가지

        #그렇게 해서 인코더와 디코더를 반환함.
        return conv_enc, conv_dec

    def _extract_patches(self, img, patch_size, stride, nonmask_point_idx, mask_point_idx):
                        # [512,32,32],      1 ,       1  ,  인덱스 저장 리스트

        n_dim = 3
        assert img.dim() == n_dim, 'image must be of dimension 3.'
        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        #kH=1, kW=1，dH, dW=1
        # img -> input_windows  = [512, 32, 32] -> [512, 32, 32, 1, 1]
        input_windows = img.unfold(1, kH, dH).unfold(2, kW, dW) # torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

        # 1: 채널수 2: 가로축 패치 개수 3: 세로축 패치 개수 4,5는 가로세로
        i_1, i_2, i_3, i_4, i_5 = input_windows.size(0), input_windows.size(1), input_windows.size(2), input_windows.size(3),input_windows.size(4)
        #input_window = [512,32,32,1,1]
        input_windows = input_windows.permute(1,2,0,3,4).contiguous().view(i_2*i_3, i_1, i_4, i_5)

        # => [1024,512,1,1] 채널 고정시키고 feature 앞으로 빼서 하나씩 분리(2차원 -> 1차원으로)
        patches_all = input_windows #1024,512,1,1  == 1,1,512,32,32 

        patches = input_windows.index_select(0, nonmask_point_idx) #It returns a new tensor, representing patches extracted from non-masked region! 마스크되지 않은 영역에서 추출된 패치를 나타내는 새로운 텐서를 반환합니다!

        maskpatches = input_windows.index_select(0,mask_point_idx)
        return patches_all, patches,maskpatches
            #   모든 패치,   nonmask 패치, mask패치


    def _extract_patches_mask(self, img, patch_size, stride, nonmask_point_idx, mask_point_idx):
        n_dim = 3
        assert img.dim() == n_dim, 'image must be of dimension 3.'
#kH=1, kW=1，dH, dW=1
        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(1, kH, dH).unfold(2, kW, dW)
        #其中i_1, i_2, i_3, i_4, i_5    ------1：通道数  2：横轴上的patch个数   3：纵轴上patch个数    4和5分别是patch的长宽
        i_1, i_2, i_3, i_4, i_5 = input_windows.size(0), input_windows.size(1), input_windows.size(2), input_windows.size(3), input_windows.size(4)
        input_windows = input_windows.permute(1,2,0,3,4).contiguous().view(i_2*i_3, i_1, i_4, i_5)
        maskpatches = input_windows.index_select(0,mask_point_idx)

        return maskpatches


