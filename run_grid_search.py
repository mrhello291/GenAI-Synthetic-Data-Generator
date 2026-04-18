import os
import subprocess
import itertools

# Define the hyperparameter grid
alphas = [0.5, 1.0, 2.0]
betas = [0.1, 0.5, 1.0]

dataname = "adult"  # Change this if testing on other datasets
gpu = 0

# Create all combinations
combinations = list(itertools.product(alphas, betas))

print(f"Starting Grid Search. Total configurations: {len(combinations)}")

# Set up the environment variable so Python knows where 'tabsyn' is
custom_env = os.environ.copy()
custom_env["PYTHONPATH"] = os.path.abspath("ctabsyn") + os.pathsep + custom_env.get("PYTHONPATH", "")

for alpha, beta in combinations:
    print(f"\n{'='*50}")
    print(f"Running VAE Training: alpha={alpha}, beta={beta}")
    print(f"{'='*50}")
    
    # Construct the command
    command = [
        "python", "ctabsyn/tabsyn/vae/main.py",
        "--dataname", dataname,
        "--gpu", str(gpu),
        "--alpha", str(alpha),
        "--beta", str(beta)
    ]
    
    # Run the command and wait for it to finish, passing the custom environment
    subprocess.run(command, env=custom_env)
    
    # Important: The VAE script saves the latent embeddings to 
    # ctabsyn/tabsyn/vae/ckpt/adult/train_z.npy
    # You need to rename this file so it isn't overwritten by the next run!
    
    ckpt_dir = f"ctabsyn/tabsyn/vae/ckpt/{dataname}"
    original_file = os.path.join(ckpt_dir, "train_z.npy")
    new_file = os.path.join(ckpt_dir, f"train_z_alpha{alpha}_beta{beta}.npy")
    
    if os.path.exists(original_file):
        os.rename(original_file, new_file)
        print(f"Saved latent space to: {new_file}")
    else:
        print(f"Error: {original_file} not found. Training may have failed.")

print("\nGrid Search Complete!")
