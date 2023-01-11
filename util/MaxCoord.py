import numpy as np
import torch
import torch.nn as nn
# Input is a tensor
# Input has to be 1*N*H*W
# output and ind: are all 0-index!!

# Additional params: pre-calculated constant tensor. `sp_x` and `px_y`.
# They are just for Advanced Indexing.
# sp_x: [0,0,..,0, 1,1,...,1, ..., 31,31,...,31],   length is 32*32, it is a list
# sp_y: [0,1,2,...,31,  0,1,2,...,31,  0,1,2,...,31]  length is 32*32, it is a LongTensor(cuda.LongTensor)
class MaxCoord():
    def __init__(self):
        pass

    def update_output(self, input, sp_x, sp_y):
        input_dim = input.dim()
        assert input.dim() == 4, "Input must be 3D or 4D(batch)."
        assert input.size(0) == 1, "The first dimension of input has to be 1!"

        output = torch.zeros_like(input)#같은 크기, 0인 텐서

        v_max,c_max = torch.max(input, 1) #input shape [1,x,32,32] => x축에서 가장 큰 값을 고름 => 결과는 [32,32]
        #이 의미는?
        #nonmask patch가 500개라고 치면 512,32,32 패치를 500,32,32패치로 만들었다.
        #이것의 의미.
        #500개의 non mask patch가 있고. 이 patch마다 32*32 feature map이 있는거니까
        #해당 patch과의 correlation value를 담고있는 map이라고 봐도 되는건가?
        #만약 그렇다고 가정했을 때, 여기서 torch.max(input,1)을 수행해서 얻어낸 32,32 feature map은 correlation score가 가장 높은 값으로 설정된다. 
        #v_max는 32*32의 각각 correlation score가 가장 높은 값이 저장되고
        #c_max는 인덱스가 저장됨. 그 인덱스가 뭐냐? x범위 내 인덱스 => 상관도가 가장 높은 채널 => 상관도가 가장높은 패치.(non mask 중에서)

        c_max_flatten = c_max.view(-1)
        v_max_flatten = v_max.view(-1)
       # output[:, c_max_flatten, sp_x, sp_y] = 1
        ind = c_max_flatten
        
        return output, ind,  v_max_flatten
