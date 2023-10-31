# Perceptron

A perceptron is a fundamental unit in artificial neural networks and is a type
of binary linear classifier. It is a mathematical model inspired by the
functioning of a biological neuron.

The perceptron **takes multiple binary input values (usually 0 or 1) and produces
a single binary output (0 or 1) based on a weighted sum of these inputs**. This
weighted sum is then passed through an activation function to produce the
output.

The primary components of a perceptron can be described as follows:

1. **Input Values $(x_1, x_2, ..., x_n)$**: These are binary values that represent the
   input features. In practical applications, continuous values may be used, but
   binary values are often employed for simplicity.

2. **Weights $(w_1, w_2, ..., w_n)$**: Each input value is associated with a weight,
   denoted as w, which represents the importance of that input. These weights
   are learned during the training process.

3. **Weighted Sum (z)**: The weighted sum of the inputs is calculated as the dot
   product of the input values and their corresponding weights:
   $z = w_1 \times x_1 + w_2 \times x_2 + ... + w_n \times x_n$.

4. **Activation Function (Step Function)**: The weighted sum is passed through an
   activation function, typically a step function. The output is 1 if the
   weighted sum is greater than or equal to a certain threshold, and 0
   otherwise.

Mathematically, the perceptron's output (y) can be expressed as:

$$y = 1,\text{ if z ≥ threshold }y = 0,\text{ if z < threshold}$$

Perceptrons are limited in their ability to model complex relationships in data
because they **can only represent linear decision boundaries**.

However, by combining multiple perceptrons in a network, more complex patterns
and decision boundaries can be learned. This laid the foundation for the
development of multilayer perceptrons and deep neural networks.
