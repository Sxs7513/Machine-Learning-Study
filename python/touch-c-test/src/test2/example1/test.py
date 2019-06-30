from ctypes import *
adder = CDLL('./adder.so')
res_int = adder.add_int(4,5)
print(res_int)