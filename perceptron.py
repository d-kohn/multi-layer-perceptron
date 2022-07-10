from numpy import *
import math

OUTPUT = 0
HIDDEN = 1

class perceptron:
	def __init__(self, input_size, bias, eta, momentum):
		self.input_size = input_size +1
		self.eta = eta
		self.momentum = momentum
		self.bias = bias
		self.weights = []
		self.delta_weights = []
		for i in range(self.input_size):
			self.weights.append(random.rand()*0.1-0.05)
			self.delta_weights.append(0.0)

	def activation(self,input):
		input[0] = self.bias
		z = dot(input,self.weights)
		return 1/(1+pow(math.e,-z))

	def calc_error_sum(self, output_error):
		sum = 0.0
		for i in range (len(output_error)):
			sum += self.weights[i]*output_error[i]
		return sum

	def update_weights(self, error, input):
		for i in range(self.input_size):
			self.delta_weights[i] = self.eta*error*input[i]+self.momentum*self.delta_weights[i]
			self.weights[i] = self.weights[i] + self.delta_weights[i]
