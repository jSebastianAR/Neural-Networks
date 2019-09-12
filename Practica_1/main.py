from TLU import *

tlu = TLU(.5)
#tlu.AND()
tlu.train_TLU("and")
print("\n\n========================================================================================\n\n")
tlu.train_TLU("or")