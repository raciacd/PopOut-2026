# Performance Measures for Message Digests, Symmetric and Asymmetric Cryptography

This project is part of the S**ecurity and Privacy** course at the **Faculty of Sciences, University of Porto**,
developed to explore and better understand common encryption and decryption methods.

## Setup and Installation

### 1. Create and Activate a Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Model Execution

### AES Counter Mode

The script uses the cryptography library to implement the method.

A benchmarking layer isolates the cryptographic operations by pre-loading data into memory, ensuring little noise in the results. The system returns runtimes in microseconds utilizing the `timeit` module to clock the execution and iterate 100 times over each file size.

Finnaly the code utilizes `utils.py` to calculate the statistics while `matplotlib` is used to generate the graphs.


Run the main file

```python
python aes_ctr_crypto.py
```

Rename the generated graph if saving multiple runs is needed. (Line 119)

```python
plot.savefig('aes_combined_plot_random_2.png', bbox_inches='tight', dpi=300) # Rename graph for every rerun
```

Alternate between run_performance_test() and run_performance_test_random() to switch between the previously generated files and newly generated random data. (Line 178)

```python
if __name__ == "__main__":
    run_performance_test() # Alternate between run_performance_test() and run_performance_test_random()
```

### RSA Hybrid Encryption
The script implements an encryption algorithm using the `cryptography` library for RSA 2048 key encapsulation and `hashlib` for SHA-256 data encapsulation via a stream cipher approach.

Similar as in the preceding AES task, a benchmarking function isolates the cryptographic computations by pre-loading the files into memory to eliminate I/O noise. The script uses the python `timeit` module to independently measure the execution of both the encryption and decryption processes, iterating 100 times over each file size and returning the results in microseconds. Finally, the code utilizes `utils.py` to calculate the statistical data (mean, 95% confidence intervals, and standard deviation), while `matplotlib` library is used to generate the comparative graphs.

Run the main file:
```bash
python rsa_crypto.py
```

Running this file will output the benchmark statistical data directly to the terminal after running the test and display a side by side linear and log-log graph comparing the encryption and decryption behaviour. The graph will also be automatically saved in the project directory as rsa_team_formatted_plot.png

### SHA-256

Benchmarking is split into two files: ```sha_crypto.py``` and ```sha_crypto_benchmark.py```. The first scripts use haslib to implement the method, which the second script then calls upon.

```sha_crypto.py``` benchmarks how long a SHA-256 hash function computation takes using repeated timing with warm-up to get stable measurements.
```sha_crypto_benchmark.py``` call the function in the previous script to benchmark the SHA-256 hash function implementation across multiple input sizes, measuring mean runtime and variability. It computes throughput (bytes per microsecond) and a 95% confidence interval, then uses Matplotlib and Pandas to plot timing vs file size on linear and log-log scales.

Run the main file:
```python
python sha_crypto_benchmark.py
```
A linear-linear and a log-log plot will be shown and a file (sha256_benchmark.png) will be saved with the same contents.

Running ```sha_crypto.py``` is possible, though not strictly necessary:
```python
python sha_crypto.py
```
Running this file will give per-file distributions of SHA-256 timing (mean, median, stddev in microseconds) without any higher-level interpretation of the listed files.
