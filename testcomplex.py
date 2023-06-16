import tensorflow as tf
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])
row_sums = tf.reduce_sum(arr, axis=1)

print(row_sums)  # [6 15]
def reshape_samples(data, num_rows, num_columns):
    data = tf.convert_to_tensor(data)
    batch_size = data.shape[0]
    reshaped_data = tf.reshape(data, (batch_size, num_rows, num_columns))


    return reshaped_data

# 生成測試數據
batch_size = 4
num_elements = 32
data = np.arange(batch_size * num_elements).reshape((batch_size, num_elements))

# 將數據重新塑形成 (8, 4)
reshaped_data = reshape_samples(data, 8, 4)

# 檢查結果
for i in range(batch_size):
    print(f"Sample {i + 1}:")
    print(reshaped_data[i])
    print(data[i])
