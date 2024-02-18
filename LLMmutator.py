import logging
import os
import chatglm_cpp
import time
import json

logger = logging.getLogger()

# get os envrionment variable
model_path = os.environ['GLM']
pipeline = chatglm_cpp.Pipeline(model_path)
# TOOLS = [
#     {
#         "name": "report_mutation",
#         "description": "Report the mutation result",
#         "parameters": {
#             "type": "object",
#             "properties": {"result": {"description": "The mutation result to report", "type": "string"}},
#             "required": ["result"],
#         },
#     },
# ]
system_prompt = '''
# 角色
你是一位测试人员。给定的输入格式，构造可能触发错误的输入。

## 可能触发错误的输入
- 分析每种数据类型的边界值。对于整数，考虑其最大值、最小值和零等。
- 对于字符串，考虑空字符串、特别长的字符串、单字符的字符串等。
- 尝试替换为错误的数据类型。

## 限制
- 只需返回一条满足条件的测试用例。
- 变异后的格式必须和原格式相同！
- 注意：不要输出任何解释，不要输出其他内容!

## 例子
### 输入1
{"raw_content": 123}
### 输出1
{"raw_content": -1}

### 输入2
{"raw_content": "test"}
### 输出2
{"raw_content": "testtesttesttest"}
'''


from itertools import combinations

def generate_combinations(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")

    length = len(list1)
    all_combinations = []

    # 枚举从第一个列表中取出元素的所有可能性
    for r in range(length + 1):
        for indices in combinations(range(length), r):
            combo = []
            for i in range(length):
                if i in indices:
                    combo.append(list1[i])
                else:
                    combo.append(list2[i])
            all_combinations.append(combo)

    return all_combinations

def LLMmutator(str_list):
    tmp = []
    for t in str_list:
        text = t
        if isinstance(t, bytes):
            text = t.decode()
        start = time.time()
        history = []
        res = chat(text, history)
        end = time.time()
        logger.info("[Info] Chat: %s Time: %f", res, end - start)

        if isinstance(t, bytes):
            tmp.append(res.encode())
        else:
            tmp.append(res)
    res = generate_combinations(tmp, str_list)
    logger.info("[Info] LLMmutator result: %s", res)
    return res

def chat(text, history=[]):
    h = [chatglm_cpp.ChatMessage(role="system", content=system_prompt)]
    if history != []:
        h.extend(history)
    try:
        json.loads(text)
        h.append(chatglm_cpp.ChatMessage(role="user", content=text))
    except Exception as e:
        text = json.dumps({"raw_content": text})
        h.append(chatglm_cpp.ChatMessage(role="user", content=text))
    res = pipeline.chat(h, max_new_tokens=max(50, len(text)*2))
    logger.info("[Info] Chat result: %s", res.content)
    try:
        r = json.loads(res.content)
        return r["raw_content"]
    except:
        return text


# test
if __name__ == "__main__":
    print(LLMmutator([b'hello', b'world']))
    print(LLMmutator([b'hell111o', b'world222']))