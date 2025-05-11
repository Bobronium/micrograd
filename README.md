
# micrograd

![awww](puppy.jpg)

A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

### Installation

```bash
pip install micrograd
```

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from micrograd.engine import Value

weight_param = Value(-4.0)
bias_term = Value(2.0)
linear_combination = weight_param + bias_term
nonlinear_component = weight_param * bias_term + bias_term**3
linear_combination += linear_combination + 1
linear_combination += 1 + linear_combination + (-weight_param)
nonlinear_component += nonlinear_component * 2 + (bias_term + weight_param).relu()
nonlinear_component += 3 * nonlinear_component + (bias_term - weight_param).relu()
feature_difference = linear_combination - nonlinear_component
squared_error = feature_difference**2
loss_component = squared_error / 2.0
final_loss = loss_component + 10.0 / squared_error
print(f'{final_loss.data:.4f}') # prints 24.7041, the outcome of this forward pass
final_loss.backward()
print(f'{weight_param.grad:.4f}') # prints 138.8338, i.e. the numerical value of d(final_loss)/d(weight_param)
print(f'{bias_term.grad:.4f}') # prints 645.5773, i.e. the numerical value of d(final_loss)/d(bias_term)
```

### Training a neural net

The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `micrograd.nn` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neuron](moon_mlp.png)

### Tracing / visualization

For added convenience, the notebook `trace_graph.ipynb` produces graphviz visualizations. E.g. this one below is of a simple 2D neuron, arrived at by calling `draw_dot` on the code below, and it shows both the data (left number in each node) and the gradient (right number in each node).

```python
from micrograd.nn import Neuron
from micrograd.engine import Value
from micrograd.trace_graph import draw_dot

neuron_2d = Neuron(2)
input_features = [Value(1.0), Value(-2.0)]
neuron_output = neuron_2d(input_features)
computation_graph = draw_dot(neuron_output)
```

![2d neuron](gout.svg)

### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python -m pytest
```

### License

MIT
