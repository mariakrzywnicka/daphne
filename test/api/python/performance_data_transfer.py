import numpy as np
import time
import matplotlib.pyplot as plt
from daphne.context import DaphneContext
import tensorflow as tf

# Initialize Daphne context
dctx = DaphneContext()

# Define matrix/vector sizes for benchmarks
sizes = [100, 500, 1000]

# Define number of repetitions for averaging
repetitions = 10

# Results dictionary
results = {
    "operation": [],
    "size": [],
    "numpy": [],
    "tensorflow": [],
    "daphnelib": []
}

# Benchmark scenarios
def benchmark_operation(operation_name, size, numpy_func, tensorflow_func, daphnelib_func):
    # Generate random data
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    # NumPy
    numpy_times = []
    for _ in range(repetitions):
        start = time.time()
        numpy_func(A, B)
        end = time.time()
        numpy_times.append(end - start)
    numpy_avg = np.mean(numpy_times)

    # TensorFlow
    tf_A = tf.constant(A)
    tf_B = tf.constant(B)
    tensorflow_times = []
    for _ in range(repetitions):
        start = time.time()
        tensorflow_func(tf_A, tf_B)
        end = time.time()
        tensorflow_times.append(end - start)
    tensorflow_avg = np.mean(tensorflow_times)

    # DaphneLib
    daphne_A = dctx.from_numpy(A, shared_memory=True)
    daphne_B = dctx.from_numpy(B, shared_memory=True)
    daphnelib_times = []
    for _ in range(repetitions):
        start = time.time()
        daphnelib_func(daphne_A, daphne_B)
        end = time.time()
        daphnelib_times.append(end - start)
    daphnelib_avg = np.mean(daphnelib_times)

    # Store results
    results["operation"].append(operation_name)
    results["size"].append(size)
    results["numpy"].append(numpy_avg)
    results["tensorflow"].append(tensorflow_avg)
    results["daphnelib"].append(daphnelib_avg)

# Define operations
def numpy_matmul(A, B):
    return A @ B

def tensorflow_matmul(A, B):
    return tf.matmul(A, B)

def daphnelib_matmul(A, B):
    return (A @ B).compute()

def numpy_add(A, B):
    return A + B

def tensorflow_add(A, B):
    return tf.add(A, B)

def daphnelib_add(A, B):
    return (A + B).compute()

def benchmark_string_transfer(size):
    # Generate random string data
    strings = [f"string_{i}" for i in range(size)]

    # Convert to a 2D array (each string in its own row)
    strings_array = np.array(strings, dtype=object).reshape(-1, 1)

    # Transfer to DaphneLib
    start = time.time()
    daphne_strings = dctx.from_numpy(strings_array, shared_memory=False)
    transfer_to_daphne_time = time.time() - start

    # Store results
    results["operation"].append("String Transfer")
    results["size"].append(size)
    results["numpy"].append(transfer_to_daphne_time)  # Using NumPy column for transfer to Daphne
    results["tensorflow"].append(0)  # TensorFlow does not support string transfer
    results["daphnelib"].append(0)  # DaphneLib does not support string transfer

    print(f"String transfer for size {size}:")
    print(f"  Transfer to DaphneLib: {transfer_to_daphne_time:.6f} seconds")

# Run benchmarks for different operations and sizes
for size in sizes:
    print(f"Running benchmarks for size: {size}x{size}")
    benchmark_operation("Matrix Multiplication", size, numpy_matmul, tensorflow_matmul, daphnelib_matmul)
    benchmark_operation("Matrix Addition", size, numpy_add, tensorflow_add, daphnelib_add)
    benchmark_string_transfer(size) 

# End-to-End Experiment: Linear Regression
def end_to_end_linear_regression(size):
    print(f"\nRunning end-to-end linear regression for size: {size}")
    # Generate random dataset
    X = np.random.rand(size, 10)
    y = np.random.rand(size)

    # Transfer to DaphneLib
    start = time.time()
    daphne_X = dctx.from_numpy(X, shared_memory=True)
    daphne_y = dctx.from_numpy(y, shared_memory=True)
    transfer_time = time.time() - start

    # Perform linear regression manually
    start = time.time()
    X_transpose = daphne_X.t()  # Transpose of X
    X_transpose_X = (X_transpose @ daphne_X).compute()  # X^T X
    X_transpose_y = (X_transpose @ daphne_y).compute()  # X^T y
    beta = np.linalg.solve(X_transpose_X, X_transpose_y)  # Solve for beta
    computation_time = time.time() - start

    # Transfer results back to Python
    start = time.time()
    result_numpy = beta  # Already in NumPy format
    result_transfer_time = time.time() - start

    print(f"Data transfer to DaphneLib: {transfer_time:.6f} seconds")
    print(f"Computation in DaphneLib: {computation_time:.6f} seconds")
    print(f"Result transfer to Python: {result_transfer_time:.6f} seconds")

# Run end-to-end experiment
end_to_end_linear_regression(1000)

# Print results
print("\nBenchmark Results:")
print(f"{'Operation':<20}{'Size':<10}{'NumPy (s)':<15}{'TensorFlow (s)':<15}{'DaphneLib (s)':<15}")
print("-" * 75)
for i in range(len(results["operation"])):
    print(f"{results['operation'][i]:<20}{results['size'][i]:<10}{results['numpy'][i]:<15.6f}{results['tensorflow'][i]:<15.6f}{results['daphnelib'][i]:<15.6f}")