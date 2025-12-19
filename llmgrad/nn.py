import ctensor
import numpy as np
try:
    from safetensors.numpy import save_file, load_file
except ImportError:
    save_file, load_file = None, None

class Module:
    def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
    def forward(self, x): raise NotImplementedError
    def parameters(self): return []
    def zero_grad(self):
        for p in self.parameters(): p.zero_grad()
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def save_pretrained(self, filename):
        if save_file is None: raise ImportError("pip install safetensors")
        tensors_dict = {}
        for name, param in self.named_parameters().items():
            arr = np.array(param.data, dtype=np.float32).reshape(param.shape)
            tensors_dict[name] = arr
        save_file(tensors_dict, filename)
        print(f"Model saved to {filename}")

    def load_pretrained(self, filename):
        if load_file is None: raise ImportError("pip install safetensors")
        tensors_dict = load_file(filename)
        for name, param in self.named_parameters().items():
            if name in tensors_dict:
                arr = tensors_dict[name]
                if list(arr.shape) != param.shape:
                    print(f"Skip {name}: shape mismatch")
                    continue
                param.data = arr.flatten().tolist()
        print(f"Model loaded from {filename}")
    
    def named_parameters(self, prefix=""):
        params = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ctensor.Tensor):
                params[f"{prefix}{k}"] = v
            elif isinstance(v, Module):
                params.update(v.named_parameters(f"{prefix}{k}."))
            elif isinstance(v, list):
                for i, layer in enumerate(v):
                    if isinstance(layer, Module):
                        params.update(layer.named_parameters(f"{prefix}{k}.{i}."))
        return params

class Linear(Module):
    def __init__(self, in_f, out_f):
        self.in_f = in_f
        self.out_f = out_f
        self.weight = ctensor.Tensor.randn(in_f, out_f)
        self.bias = ctensor.Tensor.zeros(1, out_f)
    
    def forward(self, x): 
        return x.matmul(self.weight).add(self.bias)
    
    def parameters(self): return [self.weight, self.bias]
    
    def __repr__(self):
        return f"Linear(in_features={self.in_f}, out_features={self.out_f})"

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.num = num_embeddings
        self.dim = embedding_dim
        self.weight = ctensor.Tensor.randn(num_embeddings, embedding_dim)
        
    def forward(self, x):
        if not x: return None
        
        if isinstance(x[0], list):
            block_size = len(x[0])
            flat = [val for sublist in x for val in sublist]
            return ctensor.embedding_concat(self.weight, flat, block_size)
        
        return ctensor.embedding_forward(self.weight, x)
    
    def parameters(self): return [self.weight]
    
    def __repr__(self):
        return f"Embedding(num_embeddings={self.num}, embedding_dim={self.dim})"

class LayerNorm(Module):
    def __init__(self, features, eps=1e-5):
        self.features = features
        self.gamma = ctensor.Tensor.from_list([features], [1.0]*features)
        self.beta = ctensor.Tensor.from_list([features], [0.0]*features)
        self.eps = eps
        
    def forward(self, x):
        return x.layernorm(self.gamma, self.beta, self.eps)
        
    def parameters(self): return [self.gamma, self.beta]
    
    def __repr__(self):
        return f"LayerNorm(features={self.features}, eps={self.eps})"

class Tanh(Module):
    def forward(self, x): return x.tanh()

class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def parameters(self):
        p = []
        for l in self.layers: p.extend(l.parameters())
        return p
    
    def __repr__(self):
        layers_str = "\n  ".join([str(l) for l in self.layers])
        return f"Sequential(\n  {layers_str}\n)"
