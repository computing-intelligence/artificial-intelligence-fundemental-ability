from collections import defaultdict
from functools import wraps
from icecream import ic


original_price = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30, 33]
price = defaultdict(int)
for i, p in enumerate(original_price):
    price[i+1] = p


def memo(func):
    cache = {}
    @wraps(func)
    def _wrap(n): ## ? *args, **kwargs
        if n in cache: result = cache[n]
        else:
            result = func(n)
            cache[n] = result
        return result
    return _wrap


@memo
def r(n):
    max_price, split_point = max(
        [(price[n], 0)] + [(r(i) + r(n-i), i) for i in range(1, n)], key=lambda x: x[0]
    )
    solution[n] = (split_point, n - split_point)
    
    return max_price


def not_cut(split): return split == 0

def parse_solution(target_length, revenue_solution):
    left, right = revenue_solution[target_length]
    
    if not_cut(left): return [right]

    return parse_solution(left, revenue_solution) + parse_solution(right, revenue_solution)


solution = {}
r(50)
ic(parse_solution(20, solution))
ic(parse_solution(19, solution))
ic(parse_solution(27, solution))
