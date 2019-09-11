import numpy as np
import matplotlib as mp

class TLU(object):
	"""docstring for TLU"""
	def __init__(self, learning_rate):
		self.learning_rate, = learning_rate,
		self.weigths = False
		self.bias = False
		self.AND_vector = np.array([[0,0],[0,1],[1,0],[1,1]])
		self.OR_vector = np.array([[0,0],[0,1],[1,0],[1,1]])
		self.NOT_vector = np.array([[0,0],[0,1],[1,0],[1,1]])
		self.threshold = 1

	

	def AND(self):
		self.weights = np.random.rand(1,2) #random inialized array for weights wiht a value between 0 and 1
		self.bias = np.random.rand(1,1)#np.random.rand(2,2) #random inialized array for weights wiht a value between 0 and 1
		flag = False #flag which finish the loop when there is no change in the weights
		print(self.bias)
		print(self.weights)
		#print(np.matmul(self.weights,self.bias.transpose()))
		
		x = 0
		while(flag==False):
			#First we need to multiply the inouts vector and the weights vector
			flag = True
			for i in range(0,4):
				mul = np.add(np.matmul(self.weights,self.AND_vector[i]),self.bias)
				final_result = self.compare(mul)
				expected_result = self.AND_vector[i][0] and self.AND_vector[i][0]
				print("final_result: {} expected_result: {}".format(final_result,expected_result))
				if(final_result!=expected_result):
					self.updateTLU(expected_result,final_result,self.AND_vector[i][:])
					flag = False
				input()
			x = x+1
			
			print("epoca {} terminada".format(x))

		print(self.weights)

	def updateTLU(self,expected_result,result,input_vector):
		print("Before {} expected_result: {} final_result: {} input_vector: {}".format(self.weights,expected_result,result,input_vector))
		self.weights = np.add(self.weights,(self.learning_rate*(expected_result-result))*input_vector)
		print("After {} expected_result: {} final_result: {} input_vector: {}".format(self.weights,expected_result,result,input_vector))

	def compare(self,mul_result):
		if mul_result >= self.threshold:
			return 1
		else:
			return 0

