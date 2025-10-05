from data.loader import load_csv
from processing.regresnaAnalyza import multi_model_chain_predict
from ui.cli import get_correlation_method, get_alpha, get_user_input_columns, get_path_finding_method, get_frac, \
    get_max_depth, get_plot_palette
from processing.orezavanie import zero_diagonal, apply_sigma_mask, modify_pruned_matrix
from processing.identifikaciaRetazcov import run_selected_path_finding_method
from processing.korelacnaAnalyza import compute_correlation_matrix
from output.korelacnaMatica import save_heatmap
from output.korelacnyRetazec import save_correlation_chains
import pandas as pd

# Ensure pandas doesn't truncate the output
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.width', None)  # allow wide console output
pd.set_option('display.max_colwidth', None)  # no truncation of cell contents

# Prevent scientific notation
pd.set_option('display.float_format', lambda x: f'{x:.2f}')


def main():
    df, file = load_csv()

    # Correlation analysis
    corr_method = get_correlation_method()
    matrix = compute_correlation_matrix(df, corr_method)
    print("Correlation matrix computed. Preview:")
    print(matrix.round(2))
    save_heatmap(matrix, file, corr_method)
    ###
    zero_matrix = zero_diagonal(matrix)
    # print(f"zero_matrix:\n{zero_matrix}")
    alpha = get_alpha()
    print(f"Using alpha: {alpha}")
    pruned_matrix, sigma = apply_sigma_mask(zero_matrix, alpha=alpha)
    # print(pruned_matrix)
    print(f"[Sigma Threshold: {sigma:.4f}]")
    modified_pruned_matrix = modify_pruned_matrix(pruned_matrix)
    print("[Pruned Correlation Matrix — values ≥ σ]")
    print(modified_pruned_matrix.round(2))
    save_heatmap(modified_pruned_matrix, file, corr_method, sigma, alpha)

    # Correlation chains
    columns = list(modified_pruned_matrix.columns)
    in_col, out_col = get_user_input_columns(columns)
    pathf_method = get_path_finding_method()
    # Calculate path(s) and score(s)
    paths, scores = run_selected_path_finding_method(
        method=pathf_method,
        matrix=pruned_matrix.round(4),
        start=in_col,
        end=out_col)
    ###
    # If no paths were returned
    if not paths or not isinstance(paths, list) or len(paths) == 0:
        print(f"⚠️No path found from '{in_col}' to '{out_col}' — nodes may be disconnected.")
        return
    ###
    # If multiple paths exist, pick the one with the highest score
    if len(paths) > 1 and scores:
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_path = paths[best_idx]
        best_score = scores[best_idx]
    else:
        best_path = paths[0]
        best_score = scores[0] if scores else sum(abs(matrix.loc[a, b]) for a, b in zip(paths[0], paths[0][1:]))

    # Regression analysis
    frac = get_frac()
    max_depth = get_max_depth()
    palette = get_plot_palette()
    # Run regression on the best path
    df_with_predictions, error_metrics = multi_model_chain_predict(df, path=best_path, frac=frac, max_depth=max_depth)
    ###
    # Save visualization of the best chain with outputs from regression analysis
    save_correlation_chains(
        matrix=matrix,
        paths=[(best_path, best_score)],
        file_name=file,
        method=corr_method,
        alpha=alpha,
        sigma=sigma,
        path_finding_method=pathf_method,
        start_node=in_col,
        end_node=out_col,
        error_metrics=error_metrics,
        palette=palette
    )


if __name__ == "__main__":
    main()
