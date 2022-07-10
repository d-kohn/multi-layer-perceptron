from copy import deepcopy
from numpy import *

class perceptron_output:
	def __init__(self,input_size,bias,target):
		self.input_size = input_size
		self.target = target
		self.bias = bias
		self.weights = []
		for i in range(input_size+1):
			self.weights.append(random.rand()*0.1-0.05)

	def train(self,input,target,eta):
		success = 0
		self.output = dot(input,self.weights)
		if self.output > 0:
			activation = 1
		else:
			activation = 0
		
		if activation != target:
			for i in range(len(self.weights)):
				self.weights[i] += eta*(target-activation)*input[i]
		else:
			success = 1
		
		return success

	def test(self,input):		
		if dot(input,self.weights) > 0:
			return 1
		else:
		 	return 0
		
		# if activation == target:
		# else:
		# 	success = 1
		
		# return success


class perceptron_hidden:
	def __init__(self,input_size,bias):
		self.input_size = input_size
		self.bias = bias
		self.weights = []
		for i in range(input_size+1):
			self.weights.append(random.rand()*0.1-0.05)

	def train(self,input,target,eta):
		success = 0
		data.insert(0, self.nData)
		self.output = dot(data,self.weights)
		if self.output > 0:
			activation = 1
		else:
			activation = 0
		
		if activation != target:
			for i in range(len(self.weights)):
				self.weights[i] += eta*(target-activation)*data[i]
		else:
			success = 1
		
		return success
