from typing import List, Tuple

class BPETokenizer():
    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.id_to_char = {}
        self.char_to_id = {}

    def train(self, input_texts, vocab_size, verbose=False):
        '''
        BPE算法的训练过程
        :param input_texts: 输入语料库
        :param vocab_size: 目标构建的词典的大小
        :return: 
        '''
        # 1.对输入语料进行切分
        unique_chars = sorted( list(set(list(input_texts))))
        # 合并次数
        num_merges = vocab_size - len(unique_chars)
        merge = {}
        
        # 2.得到一个初始化的字典
        id_to_char = {idx: char for idx, char in enumerate(unique_chars)}
        char_to_id = {char: idx for idx, char in enumerate(unique_chars)}
        
        # 3.利用字典对输入语料进行id化
        ids = [char_to_id[ch] for ch in input_texts]
        vocab_idx = len(unique_chars) - 1
        
        # 4.训练，合并子词，直到字典的大小达到vocab_size
        for i in range(num_merges):
            if len(ids) == 1:
                break
            # 统计相邻子词出现的频率
            stats = self.stats(ids)
            if verbose:
                stats_text = {(id_to_char[pair[0]], id_to_char[pair[1]]): st for pair, st in stats.items()}
                print(f'{i+1}th iteration:')
                print(ids)
                print(stats_text)

            # 找出出现频率最高的相邻子词对
            pair = max(stats, key=stats.get)
            vocab_idx += 1
            #根据当前的词典，合并ids
            ids = self.merge_ids(ids, pair, vocab_idx)
        
            merge[pair] = vocab_idx
            id_to_char[vocab_idx] = id_to_char[pair[0]] + id_to_char[pair[1]]
            char_to_id[id_to_char[pair[0]] + id_to_char[pair[1]]] = vocab_idx

            if verbose:
                print(f"merge {id_to_char[pair[0]], id_to_char[pair[1]]}->{id_to_char[vocab_idx]} at he {i + 1}th iteration")
                print("after merge, the vocabulary is")
                print(id_to_char)

        self.merge = merge
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char

    def stats(self, ids: List[int], counts=None):
        '''
        统计相邻子词出现的频率
        :param ids: 根据当前词典索引化后的语料库
        :param counts: 
        :return: 
        '''
        counts = {} if counts is None else counts
        for item in zip(ids[:-1], ids[1:]):
            counts[item] = counts.get(item, 0) + 1
        return counts
    
    def merge_ids(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        '''
        合并语料库里面的相邻子词对，并更新ids
        :param ids:语料库未更新前的ids
        :param pair: 当前待合并的相邻子词对
        :param idx: 当前合并的相邻子词对在词典里的id
        :return: 
        '''
        new_ids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def encode(self, text):
        '''
        将输入文本进行切分并索引化
        :param text: 
        :return: 
        '''
        # 1.对输入文本进行简单切分
        ids = [self.char_to_id[ch] for ch in text]
        # print(ids)
        # 2 利用merge词典，进行多次合并，得到最终的输出
        while len(ids) >= 2:
            stats = self.stats(ids)
            # 寻找可以融合的最小id。对于每个相邻对，merges里面找不到的时候认为对应的id为无穷大
            # 因为我们train的时候id就是从小到大生成的，所以这里也按照一样的顺序执行
            pair = min(stats, key=lambda p: self.merge.get(p, float('inf')))
            if pair not in self.merge:
                break
            vocab_idx = self.merge[pair]
            ids = self.merge_ids(ids, pair, vocab_idx)
        return ids

    def decode(self, ids):
        '''
        将索引列表转化为文本
        :param ids: 
        :return: 
        '''
        text = "".join([self.id_to_char[idx] for idx in ids])
        return text
    
if __name__ == "__main__":
    train_text = """
    hello, this is a training text. The tokenizer will split the text into words and assign an id
    to each word. This is a fantastic world.
    """

    tokenizer = BPETokenizer()
    tokenizer.train(input_texts=train_text, vocab_size=64, verbose=False)
    print(tokenizer.id_to_char)

    print("encode: ", tokenizer.encode("hello world"))

    print("decode: ", tokenizer.decode([31, 36, 16, 34, 14, 7]))





