import random

max_value = 10
min_value = -10
min_degree = 10
step_degree = 20
max_degree = 30
max_degree_polyfit = 30
num_examples = 1000
num_threads = 10

max_cache = 10000

def create_test_set(number_examples, border):
    res = [None] * number_examples
    for i in range(0, number_examples):
        res[i] = (random.uniform(0, 1) * border*2)-border

    return res

def degree_30_relu(x):
    return 1.5691913887890796e-11*x**18  - 1.041909780063636e-9*x**16 + 4.9950367570804647e-8*x**14 - 1.7249883265080151e-6*x**12 + 4.2358308788656657e-5*x**10 - 0.00072229777454636038*x**8 + 0.0082707562480076025*x**6 - 0.061815637064412185*x**4 + 0.36237384675730507*x**2 + 0.49999999999999566*x + 0.16689539099965538



