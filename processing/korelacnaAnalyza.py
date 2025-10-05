def compute_correlation_matrix(df, method='pearson'):
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation.")
    return numeric_df.corr(method=method)
