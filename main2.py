import torch
import random
import Huffman
import heapq
import matplotlib.pyplot as plt
if __name__ == '__main__':

    avg_FER_tensor = None
    probability_p = 1 / 2 #bit 발생 확률
    lenghth = 1000 # code length
    p_list = torch.linspace(0, 0.5, 20).tolist()
    simulation_num = 100
    for sim in range(simulation_num):
        FER_list = []
        for test_p in p_list:
            BSC_p = test_p
            source_A = torch.bernoulli(probability_p*torch.ones(lenghth)).type(torch.int) # (a)
            encoding_length = 4
            decoding_length = 7

            eye_mat1 = torch.eye(encoding_length).type(torch.int)
            inde_mat = torch.tensor([[1,1,0],
                                    [1,0,1],
                                    [0,1,1],
                                    [1,1,1]])
            gen_mat = torch.cat((eye_mat1,inde_mat),dim=1).type(torch.int)
            source_A.reshape(-1, 4)
            encoded_A = torch.einsum('ij,jk->ik',source_A.reshape(-1, 4),gen_mat)%2
            encoded_A = encoded_A.reshape(-1, )


            # BSC
            y = (encoded_A + torch.bernoulli(BSC_p * torch.ones(encoded_A.shape)).type(torch.int))%2

            #syndrome decoding
            y_corr = y.clone().detach().reshape(-1,7)
            eye_mat2 = torch.eye(decoding_length-encoding_length).type(torch.int)
            torch.cat(( (inde_mat.transpose(1, 0)+1)%2,eye_mat2), dim=1).type(torch.int)
            par_check_mat =torch.cat(( inde_mat.transpose(1, 0),eye_mat2), dim=1).type(torch.int)
            syndrome = torch.einsum('ij,jk->ik', y.reshape(-1, 7),par_check_mat.transpose(1, 0))%2

            for __idx,__ in enumerate(syndrome):
                for ___idx,___ in enumerate(par_check_mat.transpose(1, 0)):
                    if torch.all(__ == ___):
                        y_corr[__idx,___idx]+=1 # error correction
                        break
            y_corr = y_corr%2
            decoded_A = y_corr[:, :4]
            FER =1-sum(torch.all(source_A.reshape(-1, 4) == decoded_A,dim=1))/250
            FER_list.append(FER.item())
        if avg_FER_tensor == None:
            avg_FER_tensor=torch.tensor(FER_list)
        else:
            avg_FER_tensor += torch.tensor(FER_list)
    avg_FER_tensor = avg_FER_tensor/simulation_num
    plt.plot(p_list,avg_FER_tensor.tolist(), 'o-',color='red')
    plt.ylim([0, 1])
    plt.grid(True, linestyle=':', which='both')
    plt.savefig('simulation_result.png')
    plt.show()
    print("".join(list(map(str, source_A.tolist()))))
    print("".join(list(map(str, encoded_A.tolist()))))


