import ctensor

class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, x):
        raise NotImplementedError("You must implement forward pass")
    
    def parameters(self):
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.weight = ctensor.Tensor.randn(in_features, out_features)
        self.bias = ctensor.Tensor.zeros(1, out_features) if bias else None
        
    def forward(self, x):
        out = x.matmul(self.weight)
        if self.bias:
            out = out.add(self.bias)
        return out
        
    def parameters(self):
        return [self.weight, self.bias] if self.bias else [self.weight]

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = ctensor.Tensor.randn(num_embeddings, embedding_dim)
        
    def forward(self, input_indices, block_size=None):
        if not input_indices:
            return None

        if isinstance(input_indices[0], list):
            flat_indices = [x for sublist in input_indices for x in sublist]
            
            if block_size is None:
                block_size = len(input_indices[0])
                
            return ctensor.embedding_concat(self.weight, flat_indices, block_size)
        
        if block_size is not None and block_size > 1:
             return ctensor.embedding_concat(self.weight, input_indices, block_size)
             
        return ctensor.embedding_forward(self.weight, input_indices)
    
    def parameters(self):
        return [self.weight]

class Tanh(Module):
    def forward(self, x):
        return x.tanh()

class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
