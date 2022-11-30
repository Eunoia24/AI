class HMM:
    def __init__(self, states, symbols, start_prob, emit_prob, trans_prob, sequence):
        self.states = states # state list (hidden)
        self.symbols = symbols # symbol list (observable)
        self.start_prob = start_prob # start probability
        self.emit_prob = emit_prob # emission probability matrix (observable)
        self.trans_prob = trans_prob # state trasition probability matrix (hidden)
        self.sequence = sequence # observations


    '''Forward Algorithm'''
    def forward(self):
        alpha = {0: []}

        # 초기 확률 구하기
        for i in range(len(self.states)):
            alpha[0].append(self.start_prob[self.states[i]] * self.emit_prob[self.states[i]][self.sequence[0]])

        # alpha = {t : [probability]}
        for i in range(len(self.sequence) - 1): # len(sequence) - 1 만큼 반복
            alpha[i+1] = []
            for j in range(len(self.states)): # t에서 t+1으로 이동
                tmp = 0
                for k in range(len(self.states)): # t의 모든 state에서의 이동 확률
                    tmp += (alpha[i][k] * trans_prob[self.states[k]][self.states[j]])
                alpha[i+1].append(tmp * self.emit_prob[self.states[j]][self.sequence[i + 1]])
        
        return alpha


    '''Backward Algorithm'''
    def backward(self, std=0):
        beta = {0: [1,1]}

        # beta = {t : [probability]}
        for i in range(1, len(self.sequence)):
            beta[i] = []
            for j in range(len(self.states)):
                tmp = 0
                for k in range(len(self.states)):
                    tmp += beta[i-1][k] * self.trans_prob[self.states[j]][self.states[k]] * self.emit_prob[self.states[k]][self.sequence[-i]]
                beta[i].append(tmp)

        # probability(O)
        result = 0
        for i in range(len(self.states)):
            result += beta[len(self.sequence) - 1][i] * self.start_prob[self.states[i]] * self.emit_prob[self.states[i]][self.sequence[0]]

        if std == 0: return result # std == 0 return probability
        else: return beta # std == 1 return beta sequence


    '''Viterbi Algorithm'''
    def decode(self):
        viterbi = []
        
        # 초기 확률 및 추정 hidden state 구하기
        for i in range(len(self.states)):
            viterbi.append(self.start_prob[self.states[i]] * self.emit_prob[self.states[i]][self.sequence[0]])
        
        print('Decoding(Hidden State Sequence) : ', end='')
        print(self.states[viterbi.index(max(viterbi))], end=' ') # 예상되는 날씨 출력
        # print(f"{viterbi} = {self.states[viterbi.index(max(viterbi))]}") # 확률값 + 예상되는 날씨 출력

        # 단계별 확률 및 추정 hidden state 구하기
        for i in range(1, len(self.sequence)):
            tmp = []
            for j in range(len(self.states)):
                for k in range(len(self.states)):
                    tmp.append(viterbi[k] * trans_prob[self.states[k]][self.states[j]]) # max(a[k][j])
                viterbi.append(max(tmp) * emit_prob[self.states[j]][self.sequence[i]]) # viterbi = ↑ * observable probability
                tmp = []
            viterbi = viterbi[len(self.states):] # t+1 hidden state 추정 후 t viterbi probability 삭제
            print(self.states[viterbi.index(max(viterbi))], end=' ') # 예상되는 날씨 출력
            # print(f'{viterbi} = {self.states[viterbi.index(max(viterbi))]}') # t 시점에 해당하는 확률값 + 예상되는 날씨 출력


    '''Learning Algorithm'''
    def learn(self):
        cnt = 1
        while True:

            # Old model evaluation + Forward & Backward Sequence
            old_prob, fd, bd = sum(self.forward()[len(sequence)-1]), self.forward(), self.backward(1)

            # gamma & xi generate
            gamma, xi = {}, {}

            # gamma
            for i in range(len(self.sequence)):
                gamma_sum, gamma[i] = 0, []
                for j in range(len(self.states)):
                    gamma_sum += fd[i][j] * bd[len(self.sequence)-i-1][j]
                for j in range(len(self.states)):
                    gamma[i].append((fd[i][j] * bd[len(self.sequence)-i-1][j]) / gamma_sum)

            # xi
            for i in range(len(self.sequence) - 1):
                xi_sum, xi[i] = 0, []
                for j in range(len(self.states)):
                    for k in range(len(self.states)):
                        xi_sum += fd[i][j] * self.trans_prob[self.states[j]][self.states[k]] * self.emit_prob[self.states[k]][self.sequence[i+1]] * bd[i+1][k]
                for j in range(len(self.states)):
                    for k in range(len(self.states)):
                        xi[i].append(fd[i][j] * self.trans_prob[self.states[j]][self.states[k]] * self.emit_prob[self.states[k]][self.sequence[i+1]] * bd[i+1][k] / xi_sum)

            # HMM(λ new)
            # New initial probability
            for i in range(len(self.states)):
                self.start_prob[i] = gamma[0][i]

            # New a(i→j) probability
            for i in range(len(self.states)):
                gamma_tmp, xi_tmp = 0, 0
                for j in range(len(self.states)):
                    for k in range(len(self.sequence) - 1):
                        gamma_tmp += gamma[k][i]
                        xi_tmp += xi[k][(i * len(self.states)) + j]
                    self.trans_prob[self.states[i]][self.states[j]] = xi_tmp / gamma_tmp

            # New bi(o(j)) probability
            for i in range(len(self.states)):
                for j in range(len(self.symbols)):
                    denominator, numerator = 0, 0
                    for k in range(len(self.sequence) - 1):
                        denominator += gamma[k][i]
                        if self.sequence[k] == self.symbols[j]:
                            numerator += gamma[k][i]
                    self.emit_prob[self.states[i]][self.symbols[j]] = numerator / denominator

            # Compare to New model evaluation
            new_prob = sum(self.forward()[len(sequence)-1])
            
            # 개선 이전 모델의 evaluation, 개선 이후 모델의 evaluation
            print(f'old_prob: {old_prob}, new_prob({cnt}번째 개선 모델): {new_prob}')

            # new lambda model의 개선률이 1% 미만이 될 경우 개선 종료
            if old_prob * 1.01 > new_prob:
                print(f'\n모델 개선 종료(개선 횟수 : {cnt})\n최종확률: {new_prob}')
                break
            # new lambda model의 개선률이 1% 이상이 될 경우 개선 반복
            else:
                cnt += 1

    '''
    학습과정 설명
    1. 기존의 forward prob를 구한다. (old_prob)
    2. Baum-Welch Algorithm을 이용해 새로운 lambda set을 구한다. (gamma,xi를 활용해 구함)
    3. 새로운 forward prob를 구한다. (new_prob)
    4. 개선률이 1% 미만이 될 때까지 위 과정을 반복한다.

    ※ new a(i to j)
    (i,j == hidden states)
    ※ numerator(form : xi[t][i][j]) => ( xi[0][i][j] / sum(xi[0]) ) + ... + ( xi[t-2][i][j] / sum(xi[t-2]) )
    ※ denominator(form : gamma[t][state]) => ( gamma[0][i] / sum(gamma[0]) ) + ... + ( gamma[t-2][i] / sum(gamma[t-2]) )
    ( 둘 다 t=T-1까지이고 t는 index 0부터 시작하므로 t-2까지 진행 )

    ※ new bi(o(j))
    (i == hidden states)
    ※ numerator(form : gamma[t][state]) => sum( if ot == vt : gamma[all t][i] )
    ※ denominator(form : gamma[t][state]) => ( gamma[0][i] / sum(gamma[0]) ) + ... + ( gamma[t-2][i] / sum(gamma[t-2]) )
    ( 둘 다 t=T-1까지이고 t는 index 0부터 시작하므로 t-2까지 진행 )
    '''

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

states = ('raniy', 'sunny')
symbols = ('walk', 'shop', 'clean')

start_prob = {
    'raniy' : 0.5,
    'sunny' : 0.5
}

emit_prob = {
    'raniy' : { 'walk' : 0.1, 'shop' : 0.4, 'clean' : 0.5 },
    'sunny' : { 'walk' : 0.6, 'shop' : 0.3, 'clean' : 0.1 }
}

trans_prob = {
    'raniy' : { 'raniy' : 0.7, 'sunny' : 0.3 },
    'sunny' : { 'raniy' : 0.4, 'sunny' : 0.6 }
}

sequence = ['walk', 'shop', 'clean', 'clean', 'walk', 'walk', 'walk', 'walk', 'clean']


test = HMM(states, symbols, start_prob, emit_prob, trans_prob, sequence) # Hidden Markov Model Instance

# print(f'Forward Probability : {test.forward()}', end='\n\n') # forward sequence 각각에 대한 확률
print(f'Forward Probability : {sum(test.forward()[len(sequence)-1])}') # Forward Algorithm(Evaluate)
# print(f'Forward Probability : {test.backward(1)}', end='\n\n') # backward sequence 각각에 대한 확률
print(f'Backward Probability : {test.backward()}') # Backward Algorithm(Evaluate)
test.decode() # Viterbi Algorithm(Decode)
print('\n')
test.learn() # Learning Algorithm(HMM(λ*))