from typing import Callable
import random


class Value:
    def __init__(self, data, _children=(), _op="", label="") -> None:
        self.data = data
        self._prev = _children
        self._op = _op
        self.label = label
        self.grad = 0.0

    def __repr__(self) -> str:
        return f"Value(label={self.label}, data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), "+")

    def __sub__(self, other):
        return Value(self.data - other.data, (self, other), "-")

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), "*")

    def __gt__(self, other):
        return self.data > other.data

    def relu(self):
        if self.data > 0:
            return Value(self.data, (self,), "relu")
        else:
            return Value(0.0, (self,), "relu")

    def sq(self):
        return Value(self.data**2, (self,), "sq")

    def backprop(self):
        if self._op == "*":
            self._prev[0].grad += self._prev[1].data * self.grad
            self._prev[1].grad += self._prev[0].data * self.grad
            for child in self._prev:
                child.backprop()
        elif self._op == "+":
            self._prev[0].grad += 1.0 * self.grad
            self._prev[1].grad += 1.0 * self.grad
            for child in self._prev:
                child.backprop()
        elif self._op == "-":
            self._prev[0].grad += 1.0 * self.grad
            self._prev[1].grad += -1.0 * self.grad
            for child in self._prev:
                child.backprop()
        elif self._op == "relu":
            if self.data > 0:
                self._prev[0].grad += 1.0 * self.grad
            else:
                self._prev[0].grad += 0.0
            for child in self._prev:
                child.backprop()
        elif self._op == "sq":
            self._prev[0].grad += 2.0 * self._prev[0].data * self.grad
            for child in self._prev:
                child.backprop()
        else:
            return


def rand_vec(size):
    return [Value(random.uniform(-1, 1)) for _ in range(size)]


def sum_values(vec):
    s = Value(0)
    for v in vec:
        s += v
    return s


class Neuron:

    def __init__(self, weights, bias, activation_fn="relu"):
        self.weights = weights
        self.bias = bias
        self.acivation_fn = activation_fn

    def __repr__(self) -> str:
        return f"Perceptron({self.weights}, activation_fn={self.acivation_fn})"

    def forward(self, inputs) -> Value:
        assert len(inputs) == len(self.weights), "Number of inputs must match number of weights"
        result = sum_values(w * v for w, v in zip(self.weights, inputs)) + self.bias
        if self.acivation_fn == "relu":
            result = result.relu()
        return result


class Layer:
    def __init__(self, input_size, num_node):
        self.input_size = input_size
        self.nodes = [Neuron(rand_vec(input_size), Value(random.uniform(-1, 1))) for _ in range(num_node)]

    def __repr__(self) -> str:
        return f"Layer(input_size={self.input_size}, num_node={len(self.nodes)})"

    def forward(self, inputs):
        assert len(inputs) == self.input_size, "Number of inputs must match layer expectation"
        return [node.forward(inputs) for node in self.nodes]


class FinalLayer:
    def __init__(self, input_size) -> None:
        self.input_size = input_size
        # self.nodes = [Neuron(rand_vec(input_size), Value(random.uniform(-1, 1)), "")]
        self.nodes = [Neuron(rand_vec(input_size), Value(0.0), "")]

    def __repr__(self) -> str:
        return f"FinalLayer(input_size={self.input_size})"

    def forward(self, inputs):
        assert len(inputs) == self.input_size, "Number of inputs must match layer expectation"
        return self.nodes[0].forward(inputs)


class SimpleNN:
    def __init__(self, input_size: int, layer_sizes: list[int]) -> None:
        self.input_size = input_size
        self.layers: list[Layer] = []
        last_layer_size = input_size
        for i in range(len(layer_sizes)):
            self.layers.append(Layer(last_layer_size, layer_sizes[i]))
            last_layer_size = layer_sizes[i]
        self.layers.append(FinalLayer(last_layer_size))

    def __repr__(self) -> str:
        return f"SimpleNN(input_size={self.input_size}, num_layers={len(self.layers)})"

    def forward(self, inputs) -> Value:
        result = inputs
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def backward(self, update_fn: Callable[[Value], Value], loss: Value):
        loss.grad = 1.0
        loss.backprop()
        for layer in self.layers:
            for node in layer.nodes:
                new_weights = []
                for weight in node.weights:
                    new_weights.append(update_fn(weight))
                node.weights = new_weights
                node.bias = update_fn(node.bias)


# Taken from Andrej Karpathy's micrograd video https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=10
from graphviz import Digraph


def trace(root: Value) -> tuple[set[Value], set[Value]]:
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root: Value):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f} "
            % (
                n.label,
                n.data,
                n.grad,
            ),
            shape="record",
        )
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
