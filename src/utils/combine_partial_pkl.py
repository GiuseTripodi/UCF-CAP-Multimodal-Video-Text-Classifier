import os
import glob
import pandas as pd


def combine_pickles(base_dir, prefix, output_name):
    """
    Combine all pickle files starting with a given prefix into one final pickle.
    """
    # Find all matching files
    files = sorted(glob.glob(os.path.join(base_dir, f"{prefix}_embeddings*.pkl")))
    if not files:
        print(f"[WARNING] No files found for {prefix}")
        return

    print(f"[INFO] Found {len(files)} files for {prefix}:")
    for f in files:
        print("   ", os.path.basename(f))

    # Load and concatenate
    dfs = []
    for f in files:
        try:
            df = pd.read_pickle(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARNING] Skipping {f} due to error: {e}")

    if not dfs:
        print(f"[ERROR] No data loaded for {prefix}")
        return

    combined = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(base_dir, f"{output_name}.pkl")
    combined.to_pickle(output_path)
    print(f"[SUCCESS] Saved combined {prefix} embeddings to {output_path}")
    print(f"   Shape: {combined.shape}")


if __name__ == "__main__":
    base_dir = "/Users/user/PycharmProjects/frozen-in-time/data/UcfCap"  # change if needed

    combine_pickles(base_dir, "Train", "Train_embeddings_all")
    combine_pickles(base_dir, "Val", "Val_embeddings_all")
    combine_pickles(base_dir, "Test", "Test_embeddings_all")
