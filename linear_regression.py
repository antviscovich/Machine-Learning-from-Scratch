# Set your data points
xs = [1,2,3,4,5,6,7,8]
ys = [2.9, 3.1, 4.5, 5.0, 6.2, 7.8, 8.1, 9.0]

# Initialize parameters weight (w) and bias (b)
w, b = 0.0, 0.0

# Set learning rate
lr = 0.01

# Train the model using gradient descent
# the number of epochs is set to 1000
# the gradients dw and db are calculated for each data point and averaged over the dataset
for epoch in range(1000):
    dw, db = 0.0, 0.0 
    for x, y in zip(xs, ys):
        y_pred = w * x + b
        error = y_pred - y
        dw += 2 * error * x / len(xs)
        db += 2 * error / len(xs)
    w -= lr * dw
    b -= lr * db

print(f"Weight: {w}, Bias: {b}")
