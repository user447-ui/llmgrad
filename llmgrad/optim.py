class SGD:
    def __init__(self, params, lr=0.1):
        self.params = list(params)
        self.lr = lr
        
    def step(self):
        for p in self.params:
            p.sgd_step(self.lr)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
