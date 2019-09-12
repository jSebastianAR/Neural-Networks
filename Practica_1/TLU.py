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

	

	def AND(self):
		self.weights = np.random.rand(1,2) #random inialized array for weights wiht a value between 0 and 1
		flag = False #flag which finish the loop when there is no change in the weights
		print("Initial weights {}".format(self.weights))
		
		x = 0
		while(flag==False): #while changes in weigths were made we have to continue with train
			#First we need to multiply the inouts vector and the weights vector
			
			flag = True #First, we assume that flag it's true
			for i in range(0,4): #For each training element
				mul = np.matmul(self.weights,self.AND_vector[i]) #we do the dot product between weigths vector and the input vector
				final_result = self.Activation_Function(mul)#get the result from the activation function
				expected_result = self.AND_vector[i][0] and self.AND_vector[i][0] #get the exactly result that we want
				print("final_result: {} expected_result: {}".format(final_result,expected_result))
				if(final_result!=expected_result): #if our result is not the same that the expected result
					self.updateTLU(expected_result,final_result,self.AND_vector[i][:])#update de weights
					flag = False#there is a change of vectors thus we put in false to avoid the while loop finishes
				#input()
			x = x+1
			
			print("Epoca {} terminada".format(x))

		print("Final weights for AND Gate {}".format(self.weights))


	"""
		Do the formula for update the weights that's equal to Wt = W + lr(t-y)*X 
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
		if mul_result >= self.threshold:
			return 1
		else:
			return 0

