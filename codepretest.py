import numpy as np

import numpy as np

# 定義一個複數
z = np.complex(3, 4)

# 獲取複數的相位角
theta = np.angle(z)

# 計算 exp(i*theta)
result = np.exp(1j * theta)

print(result)

#Analog_part = np.exp(index for index in range(0,5))
#print(Analog_part)