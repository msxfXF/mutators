import os
import logging
import random
import time
import math
from Bmutator import *
from LLM.LLMmutator import *
from SplitBinText.split import Split_text_binary, Restore_text_binary
import itertools


# 获取当然文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
# 配置日志
logging.basicConfig(
    filename=os.path.join(current_dir, 'mutator.log'),
    filemode='a', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


class Mutator():
    def __init__(self)->None:
        logger.info("[Info] INIT Mutator!!!!")

        self.last_tbytes = 0
        self.last_testcase = None
        self.last_trace_bits = None
        self.llm_times = 1
        self.bilstm_times = 10
        self.bilstm_N = 10
        self.bilstm_M = 3
        self.bilstm_extra = 1
        self.mutate_times = self.get_mutate_times()
        self.all_path = []
        self.mutate_map = {}
        self.mutate_cur_map = {}
        
    def mutate(self, input_data):
        # 检查缓存
        h = hash(bytes(input_data))
        if h in self.mutate_map and h in self.mutate_cur_map:
            if self.mutate_cur_map[h] >= len(self.mutate_map[h]) - 1:
                self.mutate_cur_map[h] = 0
            self.mutate_cur_map[h] += 1
            logger.debug("[Info] Find in cache, hash: %d, pos: %d", h, self.mutate_cur_map[h])
            return self.mutate_map[h][self.mutate_cur_map[h]]

        logger.info("[Info] Start mutator, buf:%s", input_data)
        # 分割二进制和文本
        annotations, bin_list, str_list = Split_text_binary(input_data)
        logger.debug("[Debug] bin_list: %s, str_list: %s", str(bin_list), str(str_list))

        # LLM
        l_list = LLMmutator(str_list)
        # BiLSTM
        b_list = Bmutate(bin_list)
        
        # 合并
        res = []
        for l in l_list:
            for b in b_list:
                res.append(Restore_text_binary(b, l, annotations))

        self.mutate_map[h] = res
        self.mutate_cur_map[h] = 0
        logger.info("[Info] Mutate result len: %d", len(res))
        return res[0]
    
    def before_run(self, input_data):
        logger.debug("[Debug] Before run, buf:%s", input_data)
        self.last_testcase = bytes(input_data)
        
    def post_run(self, trace_bits, cov, tbytes):
        logger.debug("[Debug] Post run, trace_bits len: %d, cov: %.5f, tbytes: %d", len(trace_bits), cov, tbytes)
        
        if tbytes > self.last_tbytes:
            self.last_tbytes = tbytes
            self.last_trace_bits = trace_bits
            self.last_testcase = self.last_testcase
            logger.info("[Info] New testcase found, tbytes: %d, testcase: %s", self.last_tbytes, self.last_testcase)
            logger.info("[Info] Edit path: %s", str(FindEditPath(self.last_testcase, self.all_path)))
            
            # 查找后再加到all_path中
            self.all_path.append(self.last_testcase)


    def get_mutate_times(self):
        # LLM: origin + 1 * split_text

        # BiLSTM: origin +  C_Top10_3(120次，每次+1随机位置) * 变异数(10次)
        #（一共 2401 次）

        # tiny_havoc:
        # 1: Flip single bit
        # 2: Set byte to interesting value(-128, -1, 0, 1, 16, 32, 64, 100, 127)（9个）
        # 3: Randomly sub to byte
        # 4: Randomly add to byte
        # 5: Set a random byte to a random value


        llm_times = self.llm_times + 1

        bilstm_times = math.comb(self.bilstm_N, self.bilstm_M) * self.bilstm_times + 1
        total_times = llm_times * bilstm_times - 1
        logger.info("[Info] Total mutate times: %d", total_times) 
        return total_times
    

# test
if __name__ == "__main__":
    mutator = Mutator()
    for i in range(1000):
        print(mutator.mutate(b'Fuzz\x00\x01\x02'))

    
