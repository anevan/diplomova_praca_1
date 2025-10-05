import os
import seaborn as sns
import matplotlib.pyplot as plt


def save_heatmap(matrix, file, method=None, sigma=None, alpha=None):
    # Determine if it's a pruned or full correlation matrix
    is_pruned = sigma is not None or alpha is not None

    # Build the output directory path
    parts = ["outputs", file]
    if method:
        parts.append(method.lower())
    if is_pruned and alpha is not None:
        parts.append(f"alpha_{alpha:.2f}")
    output_dir = os.path.join(*parts)
    os.makedirs(output_dir, exist_ok=True)

    # File name: correlation_matrix.png or pruned_correlation_matrix.png
    filename = "pruned_correlation_matrix.png" if is_pruned else "correlation_matrix.png"
    output_path = os.path.join(output_dir, filename)

    # Generate plot title
    title = f"Pruned Correlation Matrix of {file}" if is_pruned else f"Correlation Matrix of {file}"
    if is_pruned and sigma is not None and alpha is not None:
        title += f" (σ = {sigma:.3f}, α = {alpha:.3f})"
    if method:
        title += f" [{method.capitalize()}]"

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Heatmap saved to: {output_path}")
