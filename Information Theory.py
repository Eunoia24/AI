import math, copy
from fractions import Fraction

########################################################################################################

# 1. 소수 → 분수 변환 함수
def approximate_fraction(decimal, max_denominator=1000):
    return Fraction(decimal).limit_denominator(max_denominator)

# 2. 엔트로피 계산 함수
def entropy(prob: list) -> float:
    return -sum([i * math.log2(i) for i in prob if i != 0])

# 3. 조건부 엔트로피 계산 함수
# if axis=0 row standard, axis=1 column standard
def conditional_entropy(prob: list, axis: int = 0) -> float:
    if not axis:
        return sum([sum(prob[i]) * entropy([j / sum(prob[i]) for j in prob[i]]) for i in range(len(prob))])
    else:
        prob = [[row[i] for row in prob] for i in range(len(prob[0]))]
        return sum([sum(prob[i]) * entropy([j / sum(prob[i]) for j in prob[i]]) for i in range(len(prob))])

########################################################################################################

prob = [[1/8, 1/16, 1/32, 1/32],
        [1/16, 1/8, 1/32, 1/32],
        [1/16, 1/16, 1/16, 1/16],
        [1/4, 0, 0, 0]]

print(approximate_fraction(conditional_entropy(prob)))