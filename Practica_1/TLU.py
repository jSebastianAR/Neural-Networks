import numpy as np
import matplotlib as mp

class TLU(object):
	"""docstring for TLU"""
	def __init__(self, learning_rate):
		self.learning_rate = learning_rate #This variable makes the training lower or faster
		self.weigths = False #Vector dor weights, initialized in False because we have to initialize in random values
		self.AND_vector = np.array([[0,0],[0,1],[1,0],[1,1]]) #Training dataset for AND Gate
		self.OR_vector = np.array([[0,0],[0,1],[1,0],[1,1]]) #Training dataset for OR Gate
		self.NOT_vector = np.array([[0,0],[0,1],[1,0],[1,1]])
		self.threshold = 1 #Umbral value used for the TLU's activation function
		self.type_gate = ""

	def train_TLU(self,gate):	
		self.weights = np.random.rand(1,2) #random inialized array for weights wiht a value between 0 and 1
		self.type_gate = gate
		flag = False #flag which finish the loop when there is no change in the weights
		print("Initial weights {}".format(self.weights))
		
		x = 0
		while(flag==False): #while changes on weigths were made, we have to continue with training
			
			flag = True #First, we assume that flag it's true
			for i in range(0,4): #For each training element
				mul = np.matmul(self.weights,self.AND_vector[i]) #we do the dot product between weigths vector and the input vector
				final_result = self.Activation_Function(mul)#get the result from the activation function
				expected_result = self.type_Gate(self.AND_vector[i][0],self.AND_vector[i][1]) #get the exactly result that we want
				print("final_result: {} expected_result: {}".format(final_result,expected_result))
				if(final_result!=expected_result): #if our result is not the same that the expected result
					self.updateTLU(expected_result,final_result,self.AND_vector[i][:])#update de weights
					flag = False#there is a change of vectors thus we put in false to avoid the while loop finishes
				#input()
			x = x+1
			
			print("Epoca {} terminada".format(x))

		print("Final weights for {} Gate {}".format(self.type_gate,self.weights))


	"""
		Does the formula for update the weights that's equal to Wt = W + lr(t-y)*X 
		Wt = new updated weights
		W = current weights
		lr = learning rate
		t = expected result
		y = current result we got from the Activation_Function
		X = current vector input what we are evaluating
	"""	
	def updateTLU(self,expected_result,result,input_vector):
		print("Before {} expected_result: {} final_result: {} input_vector: {}".format(self.weights,expected_result,result,input_vector))
		self.weights = np.add(self.weights,(self.learning_rate*(expected_result-result))*input_vector)
		print("After {} expected_result: {} final_result: {} input_vector: {}".format(self.weights,expected_result,result,input_vector))


	def Activation_Function(self,mul_result):
		if mul_result >= self.threshold: #compares the result with the umbral
			return 1
		else:
			return 0

	def type_Gate(self,value1,value2):
		if "and" == self.type_gate: #if we ask for train an AND gate
			return value1 and value2
		elif "or" == self.type_gate: #if we ask for train an OR gate
			return value1 or value2

