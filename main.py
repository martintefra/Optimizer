# main.py
from benchmarks.run_benchmark import run_benchmark

if __name__ == "__main__":
    results = run_benchmark()
    print("Benchmark Results:")
    for key, accuracy in results.items():
        print(f"Dataset: {key[0]}, Model: {key[1]}, Optimizer: {key[2]}, Accuracy: {accuracy:.2f}")
