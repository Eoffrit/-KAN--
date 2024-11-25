

from kan import *
import torch
torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model1 = KAN(width=[2,5,1], grid=3, k=3, seed=42, device=device)
from kan.utils import create_dataset
# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.cos(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, device=device)
dataset['train_input'].shape, dataset['train_label'].shape
# plot KAN at initialization
model1(dataset['train_input'])
model1.plot()
# train the model
model1.fit(dataset, opt="LBFGS", steps=50, lamb=0.001)
model1.plot()
model1 = model1.prune()
model1.plot()
model1.fit(dataset, opt="LBFGS", steps=50)
model1 = model1.refine(10)
model1.fit(dataset, opt="LBFGS", steps=50)
mode = "auto" # "manual

if mode == "manual":
    # manual mode
    model1.fix_symbolic(0,0,0,'sin')
    model1.fix_symbolic(0,1,0,'x^2')
    model1.fix_symbolic(1,0,0,'exp')
elif mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs','cos']
    model1.auto_symbolic(lib=lib)
model1.fit(dataset, opt="LBFGS", steps=50)
from kan.utils import ex_round

ex_round(model1.symbolic_formula()[0][0],4)
