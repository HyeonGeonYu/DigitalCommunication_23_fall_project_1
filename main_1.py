import torch
import random
import Huffman
import heapq
def generator_source_B(input_length):

    transition_probability00 = 3/4 # 00 : 3/4, 01 : 1/4
    transition_probability10 = 1/4 # 10 : 1/4, 11 : 3/4

    tmp_list = []
    tmp_list.append(random.randint(0, 1)) # realization with initial distribution
    for _ in range(input_length-1):
        if tmp_list[-1]==0:
            if random.random()<transition_probability00:
                tmp_list.append(0)
            else:
                tmp_list.append(1)
        else:
            if random.random()<transition_probability10:
                tmp_list.append(0)
            else:
                tmp_list.append(1)
    out_tensor = torch.tensor(tmp_list)

    return out_tensor

def Huffman_decoder(input_compressed_code,h):
    tmp_str = ""
    output_list = []
    for test_bit in input_compressed_code.tolist():
        tmp_str += str(test_bit)
        if tmp_str in h.reverse_mapping.keys():
            output_list.append(h.reverse_mapping[tmp_str].tolist())
            tmp_str = ""
    return torch.tensor(output_list)

if __name__ == '__main__':
    probability_p = 1/4
    lenghth = 1000
    source_A = torch.bernoulli(probability_p*torch.ones(lenghth)).type(torch.int) # (a)
    source_B = generator_source_B(lenghth).type(torch.int) # (b)
    encoding_length = 4
    integers = torch.arange(2)
    grids = torch.meshgrid(*([integers] * encoding_length))
    result_combinations = torch.flip(torch.stack(grids, dim=-1).reshape(-1, encoding_length).type(torch.int), (0, 1))
    result_combinations_prob = torch.prod(torch.where(result_combinations == 1, probability_p, 1 - probability_p),
                                          dim=1).tolist()
    # (c)
    h_A = Huffman.HuffmanCoding(source_A, result_combinations_prob, result_combinations, torch.tensor(range(len(result_combinations_prob))))
    compressed_A = h_A.compress()
    decompressed_A = Huffman_decoder(compressed_A,h_A).reshape(-1,)

    tmp_ratio = 0
    for __,i in enumerate(result_combinations):
        print("조합 :", tuple(i.tolist()),"코드 :","{0:7}".format(h_A.codes[tuple(i.tolist())]), "길이: ", len(h_A.codes[tuple(i.tolist())]), "확률:", "{0:.2}".format(result_combinations_prob[__]))
        tmp_ratio += len(h_A.codes[tuple(i.tolist())]) * result_combinations_prob[__]
    compression_ratio = tmp_ratio/4
    print("Compression_Ratio :","{0:.3}".format(compression_ratio))
    print(compressed_A.shape)
    print("".join(list(map(str,source_A.tolist()))))
    print("".join(list(map(str,compressed_A.reshape(-1,).tolist()))))

    #(d)
    result_combinations
    transition_num = torch.tensor([0,1,2,1,2,3,2,1,1,2,3,2,1,2,1,0])
    a =torch.tensor([0.75,0.25]*8) * 0.75**(3-transition_num) * 0.25**(transition_num) # 1이라면
    b = torch.tensor(([0.25,0.75]*8)) * 0.75**(3-transition_num) * 0.25**(transition_num) # 0이라면
    transition_matrix_source_B = torch.cat((a.repeat(8, 1),b.repeat(8, 1)),dim=0).transpose(1,0)
    initial_prob = torch.ones(16,) /2 * 0.75**(3-transition_num) * 0.25**(transition_num)
    tmp_matrix_1 = (transition_matrix_source_B-torch.eye(16))
    tmp_matrix_1[0,:] = 1
    tmp_vector_1 = torch.tensor([1]+[0]*15).reshape(-1,1)
    st_st_prob = torch.einsum("ab,bc->ac", torch.inverse(tmp_matrix_1),tmp_vector_1.type(torch.float))
    st_st_prob = st_st_prob.reshape(-1,).tolist()

    h_B = Huffman.HuffmanCoding(source_B, st_st_prob, result_combinations,
                                torch.tensor(range(len(st_st_prob))))
    compressed_B = h_B.compress()
    decompressed_B = Huffman_decoder(compressed_B,h_B).reshape(-1, )

    tmp_ratio = 0
    for __, i in enumerate(result_combinations):
        print("조합 :", tuple(i.tolist()), "코드 :", "{0:7}".format(h_B.codes[tuple(i.tolist())]), "길이: ",
              len(h_B.codes[tuple(i.tolist())]), "확률:", "{0:.2}".format(st_st_prob[__]))
        tmp_ratio += len(h_B.codes[tuple(i.tolist())]) * st_st_prob[__]

    compression_ratio = tmp_ratio/ 4
    print("Compression_Ratio :", "{0:.3}".format(compression_ratio))
    print(compressed_A.shape)
    print("".join(list(map(str, source_A.tolist()))))
    print("".join(list(map(str, compressed_A.reshape(-1, ).tolist()))))

