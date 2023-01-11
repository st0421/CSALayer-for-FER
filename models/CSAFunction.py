import torch
from util.NonparametricShift import NonparametricShift
from util.MaxCoord import MaxCoord
import util.util as util
import torch.nn as nn


from torch.autograd import Variable
class CSAFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask, shift_sz, stride, triple_w, flag, nonmask_point_idx,mask_point_idx ,flatten_offsets, sp_x, sp_y):
        assert input.dim() == 4, "Input Dim has to be 4"
        ctx.triple_w = triple_w
        ctx.flag = flag #[1024]이지만 0기반 마스킹 인덱스의 값만 1
        ctx.flatten_offsets = flatten_offsets


        ctx.bz, c_real, ctx.h, ctx.w = input.size()
        #  1      512    32       32
        c = c_real
        ctx.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor

        assert mask.dim() == 2, "Mask dimension must be 2"


        # bz is the batchsize of this GPU
        output_lst = ctx.Tensor(ctx.bz, c, ctx.h, ctx.w)
        # 1, 32, 32
        ind_lst = torch.LongTensor(ctx.bz, ctx.h*ctx.w, ctx.h, ctx.w)
        # 1,1024,32,32
        if torch.cuda.is_available:
            ind_lst = ind_lst.cuda()
            nonmask_point_idx = nonmask_point_idx.cuda()
            mask_point_idx = mask_point_idx.cuda()
            sp_x = sp_x.cuda()
            sp_y = sp_y.cuda()

        #배치사이즈는 1
        for idx in range(ctx.bz):
            inpatch = input.narrow(0, idx, 1)
            # [1,512,32,32]
            output = input.narrow(0, idx, 1)
            Nonparm = NonparametricShift()

            # _, 512->x 인코더, 1024 -> 512 디코더, _, [x,512,1,1], [1024-x,512,1,1]  (feature 값 있음)
            _, conv_enc, conv_new_dec,_,known_patch, unknown_patch = Nonparm.buildAutoencoder(inpatch.squeeze(), False, False, nonmask_point_idx,mask_point_idx,  shift_sz, stride)
            # 인자 [512,32,32], False, False, nonmask 패치의 위치를 저장한 리스트, mask 위치 저장 리스트, 1, 1

            output_var = Variable(output)
            #[1,512,32,32]
            tmp1 = conv_enc(output_var)
            #[1,x,32,32]
            #output 1,512,32,32 feature를 encoder에 넣어서 1,x,32,32를

            maxcoor = MaxCoord()
            kbar, ind, vmax = maxcoor.update_output(tmp1.data, sp_x, sp_y)# [1,753,32,32], 32, 32 -> [1,753,32,32],1024, 1024
            #vmax는 32*32의 각각 correlation score가 가장 높은 값 저장
            #ind는 상관도가 가장높은 패치(픽셀)를 의미함.
            #kbar는 input(tmp1)과 사이즈가 같은 zero 텐서 [1,x,32,32]

            real_patches = kbar.size(1) + torch.sum(ctx.flag) #non-mask 패치 개수와 mask된 패치 개수의 합 => 1024(scalar)
            #[x]+mask 패치 수 => 1024

            vamx_mask=vmax.index_select(0,mask_point_idx) #채널 별 가장 높은 값을 뽑아 모아놓은 vmax(1024)에서 마스크 인덱스 뽑아온다. 마스크 패치를 선택해서 vamx_mask에 넣고
            # vmax에서 마스크포인트의 인덱스만 뽑아와서 vamx_mask로 따로 저장.

            _, _, kbar_h, kbar_w = kbar.size() #[1,x,32,32]
            out_new = unknown_patch.clone() # [1024-x,512,1,1]  
            out_new=out_new.zero_() #unknown patch개수만큼 받아놓음 [1024-x,512,1,1] (값 없음) 
            mask_num=torch.sum(ctx.flag) #마스크 포인트 개수 저장

            in_attention=ctx.Tensor(mask_num,real_patches).zero_() #[1024-x,1024] (값 없음)
            #마스크패치에 대한 다른 패치간의 attention 값을 담는건가?

            kbar = ctx.Tensor(1, real_patches, kbar_h, kbar_w).zero_() #[1,1024,32,32] (값 없음)

            ind_laten=0
            for i in range(kbar_h):
                for j in range(kbar_w):
                    indx = i*kbar_w + j #패치 하나씩 살펴보는 중 [i][j]와 동일.
                    check=torch.eq(mask_point_idx, indx ) #마스크 인덱스와 현재 인덱스가 동일하면 check = 1
                    non_r_ch = ind[indx] #ind는 상관도가 가장 높은 인덱스를 던짐.
                    #non_r_ch는 현재 인덱스와 상관도가 가장 높은 인덱스를 얻어냄.
                    #ind의 x차원은 non-mask patch인데?
                    #그러니까 non-mask patch에서 상관도가 가장 높은 패치를 나타내는건데 . 그 나타내는 패치가 masked 일 수 있나?
           
                    offset = ctx.flatten_offsets[non_r_ch] #현재 인덱스와 상관도가 가장 높은인덱스의 offset을 가져옴
                    #offset은 간접주소잖아.

                    correct_ch = int(non_r_ch + offset) #correct_ch는 상관도 가장 높은 인덱스와 오프셋을 더해버려 이거 어따 씀?
                    #이렇게하면 non_r_ch의 non-mask 패치를 가리킴(?)

                    if(check.sum()>=1): #마스크 패치면
                        known_region=known_patch[non_r_ch] 
                        #[x, 512,32,32]에서 현재 인덱스와 상관도가 가장 높은 인덱스(=> 당연히 known feature 단)의 feature 값을 가져옴.
                        unknown_region=unknown_patch[ind_laten] #[1024-x,512,1,1]에서 첫번째 인덱스값 가져옴 [512,1,1]
                        #unknown_region을 가져온이유? 이제 여기 채워서 나중에 known region이랑 결합해야하니까
                    
                        if ind_laten==0: #첫 패치 생성 (고려할 이전 패치 없음.)
                            out_new[ind_laten]=known_region #out_new가 최종 생성 패치 리스트 인듯.
                            in_attention[ind_laten,correct_ch]=1    #correct_ch는 현재 패치와 상관도가 가장 높은 feature index이고 offset은 
                            #[1024-x,1024]
                            #첫 마스크 인덱스에 대한 attention은 correct_ch다 이걸 표현하는건가
                            #첫 마스크 인덱스에 대해 correct_ch에 1 줌
                            kbar[:, :, i, j] = torch.unsqueeze(in_attention[ind_laten], 0) #kbar [1,1024,32,32]
                        
                        elif ind_laten!=0: #마스크를 채운 전적이 있으면
                            little_value = unknown_region.clone() #[512,1,1] unkown_region가져옴 여기 coarse하게 값 있음.
                         
                            ininconv = out_new[ind_laten - 1].clone() #전에 생성한 결과물 가져온다
                            ininconv = torch.unsqueeze(ininconv, 0)

                            value_2 = little_value * (1 / (little_value.norm(2) + 1e-8)) #정규화하고
                            conv_enc_2 = nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=False)
                            value_2 = torch.unsqueeze(value_2, 0) #[1,512,1,1]
                            conv_enc_2.weight.data = value_2
                            ininconv_var = Variable(ininconv) #[1,512,1,1]
                            at_value = conv_enc_2(ininconv_var) #전에 생성한 결과물을 conv(256,1)에 집어넣음
                            #conv_enc는 현재 패치정규화시켜 가중치로 사용하는 conv를 이전패치를 입력으로 스칼라를 뽑아냄. 
                            #현재 패치와 이전 패치의 cross correlation을 뽑아냄.
                            at_value_m = at_value.data #[1,1,1,1]
                            at_value_m=at_value_m.squeeze() # scalar ex)tensor(1177.0146)
                            at_final_new = at_value_m / (at_value_m + vamx_mask[ind_laten])#Dad/(Dad+Dmax)
                            #이전패치와의 관계가 conv_enc_2로 나옴 이 값과 
                            #채널 별 상관도가 가장 높은 값을 뽑아 모아놓은 vmax(1024)에서 마스크 인덱스 뽑아온다.
                            # vmax에서 마스크포인트의 인덱스만 뽑아와서 vamx_mask로 따로 저장.
                            # 즉 vamx_mask[ind_laten]은 현재 마스크와 가장 높은 상관도값임. ex) tensor(1731.3944)

                            at_final_ori = vamx_mask[ind_laten] / (at_value_m + vamx_mask[ind_laten])#Dmax/(Dad+Dmax)
                            #현재 마스크 패치와 가장 높은 상관도와, 이전패치와의 상관도 값을 구함.
                            out_new[ind_laten] = (at_final_new) * out_new[ind_laten - 1] + (at_final_ori) * known_region
                            #위에서 구한 값과, 이전패치, 상관도 높은 패치 값을 곱해서 mi값을 추정함.
                            #근데 추정해서 out_new에 하나씩 집어넣는데 이거 안씀. 코미디임 ㅋㅋ
                            in_attention[ind_laten]=in_attention[ind_laten-1]*at_final_new.item()
                            #현재 마스크 패치에 대한 attention은 이전 패치에 대한 attention에 Dad/(Dad+Dmax)를 곱한 값해서 그대로 넣어준다.
                            in_attention[ind_laten,correct_ch]=in_attention[ind_laten,correct_ch]+at_final_ori.item()
                            #위에는 리스트에 스칼라를 곱해서 통째로 넣어줬고 여기선 진짜 스칼라 넣어줌
                            #ind_laten은 하나씩 계속 증가하니까 첫번째 마스크는 휑 하고 증가할 수록 값들이 많겠네 많이 채워져 있겠네
                            #이전 프레임의 리스트에 스칼라 곱하고 덮어씌워버린 후에 값 추가하니까
                            
                            kbar[:, :, i, j] = torch.unsqueeze(in_attention[ind_laten], 0)
                            # [1,1024]를 kbar[:,:,i,j] [1,1024,32,32]에 집어넣음
                        ind_laten+=1 #마스크 패치에 대한 연산을 할 떄마다 증가시킴 => 마스크영역을 채울 때 마다 +1
                    else:
                        kbar[:,  correct_ch , i, j] = 1
            kbar_var = Variable(kbar)   #[1,1024,32,32]
            result_tmp_var = conv_new_dec(kbar_var) #conv(1024,512)
            result_tmp = result_tmp_var.data
            output_lst[idx] = result_tmp # [512,32,32]
            ind_lst[idx] = kbar.squeeze()# [1024,32,32]


        output = output_lst [512,32,32]

        ctx.ind_lst = ind_lst
        return output



    def backward(ctx, grad_output):
        ind_lst = ctx.ind_lst


        c = grad_output.size(1)



        grad_swapped_all = grad_output.clone()

        spatial_size = ctx.h * ctx.w

        W_mat_all = Variable(ctx.Tensor(ctx.bz, spatial_size, spatial_size).zero_())
        for idx in range(ctx.bz):
            W_mat = W_mat_all.select(0, idx).clone()
            back_attention=ind_lst[idx ].clone()
            for i in range(ctx.h):
                for j in range(ctx.w):
                    indx = i * ctx.h + j
                    W_mat[indx] = back_attention[:,i,j]


            W_mat_t = W_mat.t()

            # view(c/3,-1):t() makes each line be a gradient of certain position which is c/3 channels.
            grad_swapped_weighted = torch.mm(W_mat_t, grad_swapped_all[idx].view(c , -1).t())

            # Then transpose it back
            grad_swapped_weighted = grad_swapped_weighted.t().contiguous().view(1, c , ctx.h, ctx.w)
            grad_swapped_all[idx] = torch.add(grad_swapped_all[idx], grad_swapped_weighted.mul(ctx.triple_w))

        # note the input channel and the output channel are all c, as no mask input for now.
        grad_input =grad_swapped_all

        return grad_input, None, None, None, None, None, None, None, None, None, None
