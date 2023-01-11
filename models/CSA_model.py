import torch.nn as nn
import torch
from torch.autograd import Variable
import util.util as util
from .CSAFunction import CSAFunction

class CSA_model(nn.Module):
    def __init__(self, threshold, fixed_mask, shift_sz=1, stride=1, mask_thred=1, triple_weight=1):
        super(CSA_model, self).__init__()
        #threshold=5/16
        #shift-sz=1
        #fixed_mask=1
        self.threshold = threshold
        self.fixed_mask = fixed_mask

        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.cal_fixed_flag = True # whether we need to calculate the temp varaiables this time.
        # these two variables are for accerlating MaxCoord, it is constant tensors,
        # related with the spatialsize, unrelated with mask.
        self.sp_x = None
        self.sp_y = None

    def set_mask(self, mask_global, layer_to_last, threshold):
        mask = util.cal_feat_mask(mask_global, layer_to_last, threshold)
        self.mask = mask.squeeze() #32, 32
        return self.mask

    # If mask changes, then need to set cal_fix_flag true each iteration.
    def forward(self, input):
        _, self.c, self.h, self.w = input.size()
        if self.fixed_mask and self.cal_fixed_flag == False:
            assert torch.is_tensor(self.flag), 'flag must have been figured out and has to be a tensor!'
        else:
            latter = input.narrow(0,0,1).data
            # [512,32,32]
            self.flag, self.nonmask_point_idx, self.flatten_offsets ,self.mask_point_idx= util.cal_mask_given_mask_thred(latter.squeeze(), self.mask, self.shift_sz, self.stride, self.mask_thred)
            self.cal_fixed_flag = False
            #flag는 [1024] 마스킹 위치는 1로 표시. nonmask_point_idx는 nonmask 위치 저장한 리스트
            #flatten_offsets은 non_mask와 크기가 같은 offset
            #이것들을 가지고 CSAFunction으로 들어간다.

        if not (torch.is_tensor(self.sp_x) or torch.is_tensor(self.sp_y)):
            self.sp_x, self.sp_y = util.cal_sps_for_Advanced_Indexing(self.h, self.w)

                            # [512,32,32], [32,32],         1,          1,               1,     [1024]:마스크 표시,    인덱스 저장 리스트
        return CSAFunction.apply(input, self.mask, self.shift_sz, self.stride, self.triple_weight, self.flag, self.nonmask_point_idx, self.mask_point_idx, self.flatten_offsets, self.sp_x, self.sp_y)
        # 인덱스 저장 리스트, offset(사이즈는 nonmask와 동일), 32,   32

    def __repr__(self):
        return self.__class__.__name__+ '(' \
              + 'threshold: ' + str(self.threshold) \
              + ' ,triple_weight ' + str(self.triple_weight) + ')'
