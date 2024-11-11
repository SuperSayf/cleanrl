from itertools import product

# Define the parameter space
param_space = {
    "grayscale": [True, False],
    "use_compression": [True, False],
    "resolution": [64, 74, 84],  # Combined resolution for width and height
    "frame_skip": [0, 2, 4],
}

# Separate jpeg_quality parameter
jpeg_quality = [50]

# Generate base combinations
base_experiments = list(product(*param_space.values()))

# Generate final experiments list
experiments = []
for exp in base_experiments:
    if exp[1]:  # If use_compression is True
        for quality in jpeg_quality:
            experiments.append(exp[:2] + (quality,) +
                               (exp[2], exp[2]) + exp[3:])
    else:
        # Set to 100 instead of None
        experiments.append(exp[:2] + (100,) + (exp[2], exp[2]) + exp[3:])

# Print number of generated experiments
print(f"Number of generated experiments: {len(experiments)}")

# Write experiments to a file, repeating each 3 times
with open("all_experiments.txt", "w") as f:
    f.write(
        "grayscale,use_compression,jpeg_quality,resolution_width,resolution_height,frame_skip\n"
    )
    for exp in experiments:
        for _ in range(3):
            f.write(",".join(map(str, exp)) + "\n")

print(
    "All experiments have been written to 'all_experiments.txt' (each repeated 3 times)"
)

# Remove the printing of the first 10 experiments
