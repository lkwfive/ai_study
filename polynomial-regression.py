# 多项式回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

y = np.array([3,1,5,1,1,2,2,3,5,5,4,4,5,2,6,4,4,3,4,0,8,8,9,4,5,9,5,9,8,32,8,11,11,14,12,8,5,14,10,10,22,35,9,11,11,15,12,13,18,17,16,7,5,17,15,0,17,10,15,24,16,11,24,16,12,16,22,8,18,25,21,18,16,32,27,25,20,24,24,29,36,14,19,18,29,91,36,32,94,28,22,29,31,33,40,31,32,32,29,43,39,49,58,45,71,75,80,93,118,114,90,65,46,30,23,30])
x = np.arange(y.size)

model = np.poly1d(np.polyfit(x, y, 3))
line = np.linspace(1, y.size, 120)

predict_value = model(y.size+1) # 未来一天的销量
r = r2_score(y, model(x)) # 拟合度

#预测未来10天销量
# for i in range(10):
# 	print(model(y.size+i))
# 	pass

print("predict value:", predict_value,"r value:",r)

plt.scatter(x, y)
plt.plot(line, model(line))
plt.show()