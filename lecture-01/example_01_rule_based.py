import random 
from icecream import ic


#rules = """
#复合句子 = 句子 , 连词 句子
#连词 = 而且 | 但是 | 不过
#句子 = 主语 谓语 宾语
#主语 = 你| 我 | 他 
#谓语 = 吃| 玩 
#宾语 = 桃子| 皮球
#    
#"""

rules = """
复合句子 = 句子 , 连词 复合句子 | 句子
连词 = 而且 | 但是 | 不过
句子 = 主语 谓语 宾语
主语 = 你| 我 | 他 
谓语 = 吃| 玩 
宾语 = 桃子| 皮球
    
"""

def get_grammer_by_description(description):
    rules_pattern = [r.split('=') for r in description.split('\n') if r.strip()]
    target_with_expend = [(t, ex.split('|')) for t, ex in rules_pattern]
    grammer = {t.strip(): [e.strip() for e in ex] for t, ex in target_with_expend}

    return grammer

#generated = [t for t in random.choice(grammer['句子']).split()]

#test_v = [t for t in random.choice(grammer['谓语']).split()]


def generate_by_grammer(grammer, target='句子'):
    if target not in grammer: return target

    return ''.join([generate_by_grammer(grammer, t) for t in random.choice(grammer[target]).split()])

if __name__ == '__main__':

    grammer = get_grammer_by_description(rules)

    #ic(generated)
    #ic(test_v)
    #ic(generate_by_grammer(grammer))
    ic(generate_by_grammer(grammer, target='复合句子'))


        

