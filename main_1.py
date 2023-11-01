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

def Huffman_decoder(input_compressed_code):
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

    # (c)

    probability_p
    encoding_length = 4
    integers = torch.arange(2)
    grids = torch.meshgrid(*([integers] * encoding_length))
    result_combinations = torch.stack(grids, dim=-1).reshape(-1, encoding_length).type(torch.int)
    result_combinations_prob = torch.prod(torch.where(result_combinations==1,probability_p,1-probability_p),dim=1).tolist()
    h = Huffman.HuffmanCoding(source_A, result_combinations_prob, result_combinations, torch.tensor(range(len(result_combinations_prob))))
    compressed_A = h.compress()
    decompressed_A = Huffman_decoder(compressed_A).reshape(-1,)

    tmp = 0
    for __,i in enumerate(result_combinations):
        print("조합 :", tuple(i.tolist()),"코드 :","{0:7}".format(h.codes[tuple(i.tolist())]), "길이: ", len(h.codes[tuple(i.tolist())]), "확률:", "{0:.2}".format(result_combinations_prob[__]))
        tmp += len(h.codes[tuple(i.tolist())]) * result_combinations_prob[__]
    compression_ratio = tmp/4
    print("Compression_Ratio :","{0:.3}".format(compression_ratio))
    print(compressed_A.shape)
    print("".join(list(map(str,source_A.tolist()))))
    print("".join(list(map(str,compressed_A.reshape(-1,).tolist()))))
