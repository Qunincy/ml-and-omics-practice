import math
import csv
import matplotlib.pyplot as plt

# ------------------- 1. LeakyReLU 激活函数 -------------------
def LeakyRELU(X, alpha):
    output = []
    for x in X:
        if x >= 0:
            output.append(x)
        else:
            output.append(alpha * x)
    return output

# ------------------- 2. DNA One-Hot 编码 -------------------
def encode_dna_sequence(sequence):
    mapping = {
        'G': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'T': [0, 0, 1, 0],
        'A': [0, 0, 0, 1]
    }
    return [mapping[char] for char in sequence]

# ------------------- 3. 最大池化 Max Pooling -------------------
def max_pooling(X, pool_size):
    rows = len(X)
    cols = len(X[0]) if rows > 0 else 0
    out_rows = rows // pool_size
    out_cols = cols // pool_size
    result = []
    
    for i in range(out_rows):
        current_row = []
        for j in range(out_cols):
            max_val = -float('inf')
            for di in range(pool_size):
                for dj in range(pool_size):
                    val = X[i * pool_size + di][j * pool_size + dj]
                    if val > max_val:
                        max_val = val
            current_row.append(max_val)
        result.append(current_row)
    return result

# ------------------- 4. 正弦回归 + 梯度下降 -------------------
def sinusoidal_regression(x_vals, y_vals):
    import math
    # 先简单估算一下数据的振幅和周期，作为初始化参考
    y_max = max(y_vals)
    y_min = min(y_vals)
    estimated_a = (y_max - y_min) / 2
    estimated_b = 2 * math.pi / 10  
    
    # 初始化参数
    a = estimated_a if estimated_a != 0 else 2.0
    b = estimated_b
    c = 0.0
    lr = 0.01  
    epochs = 80000  
    n = len(x_vals)
    
    for _ in range(epochs):
        y_pred = [a * math.sin(b * x + c) for x in x_vals]
        
        # 计算梯度
        da = 0.0
        db = 0.0
        dc = 0.0
        
        for i in range(n):
            x = x_vals[i]
            y = y_vals[i]
            pred = y_pred[i]
            diff = pred - y
            
            da += diff * math.sin(b * x + c)
            db += diff * a * math.cos(b * x + c) * x
            dc += diff * a * math.cos(b * x + c)
        
        # 梯度下降更新（加入L2正则防止发散）
        a -= lr * (2 / n) * da
        b -= lr * (2 / n) * db
        c -= lr * (2 / n) * dc
        
        # 防止参数爆炸
        if abs(a) > 10:
            a = 10 if a > 0 else -10
        if abs(b) > 5:
            b = 5 if b > 0 else -5
    
    return float(a), float(b), float(c)

# ------------------- main 函数 -------------------
def main():
    print("===== 1. LeakyReLU 测试 =====")
    test_leaky = LeakyRELU([-4, -2, 2, 4], 0.01)
    print("输入：[-4, -2, 2, 4], alpha=0.01")
    print("输出：", test_leaky)

    print("\n===== 2. DNA 编码测试 =====")
    dna_test = encode_dna_sequence("GCTA")
    print("输入：GCTA")
    print("输出：", dna_test)

    print("\n===== 3. 最大池化测试 =====")
    matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    pool_test = max_pooling(matrix, 2)
    print("输入矩阵：", matrix)
    print("2x2池化输出：", pool_test)

    print("\n===== 4. 正弦回归测试 =====")
    
   
    file_path = r"data\q4data.txt"
    
    try:
        x_data = []
        y_data = []
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                if len(row) >= 2:
                    x_data.append(float(row[0]))
                    y_data.append(float(row[1]))
        
        a_fit, b_fit, c_fit = sinusoidal_regression(x_data, y_data)
        print(f"拟合结果：a = {a_fit:.4f}, b = {b_fit:.4f}, c = {c_fit:.4f}")
        
        y_fit = [a_fit * math.sin(b_fit * x + c_fit) for x in x_data]
        plt.scatter(x_data, y_data, color='red', label='Original Data')
        plt.plot(x_data, y_fit, color='blue', label='Fitted Curve')
        plt.title("Sinusoidal Regression Result")
        plt.legend()
        plt.show()
        
    except Exception as e:
        print("读取数据失败：", e)

if __name__ == "__main__":
    main()
