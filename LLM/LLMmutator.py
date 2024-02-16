import logging
import os
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

def LLMmutator(str_list):
    res = [str_list]
    res.append(str_list)
    return res