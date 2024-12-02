import subprocess
from itertools import product
import json

param_grid = {
    "learning_rate": [1e-5, 1e-4, 1e-3],
    "batch_size": [8, 16, 32],
    "num_prototypes": [3, 5, 10],
    "use_context": [True, False],
}

grid_combinations = list(product(*param_grid.values()))

script_path = "PAL.py"
results = []

for idx, params in enumerate(grid_combinations):
    lr, batch_size, num_prototypes, use_context = params

    print(f"Running grid search {idx + 1}/{len(grid_combinations)}")
    print(f"Params: lr={lr}, batch_size={batch_size}, "
          f"num_prototypes={num_prototypes}, use_context={use_context}")

    # Construct command
    cmd = [
        "python", script_path,
        "--train",
        "--dataset", "prism",
        "--learning_rate", str(lr),
        "--batch_size", str(batch_size),
        "--num_prototypes", str(num_prototypes),
        "--output_model", f"model_lr{lr}_bs{batch_size}_proto{num_prototypes}_context{use_context}.pt",
        "--predict"
    ]
    if use_context:
        cmd.append("--use_context")

    print(f"Command to execute: {' '.join(cmd)}")

    # Run the command and capture output
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Command output:\n{result.stdout}")

    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e.stderr}")
        results.append({
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_prototypes": num_prototypes,
            "use_context": use_context,
            "accuracy": None,
            "error": e.stderr,
        })

with open("grid_search_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Grid search completed. Results saved to grid_search_results.json.")
