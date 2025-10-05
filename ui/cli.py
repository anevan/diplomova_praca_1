def get_user_input_columns(columns):
    print("\nFeed backward construction of correlation chains in multidimensional datasets")
    print("Available attributes:", columns)

    while True:
        x = input("Select source attribute: ").strip()
        if x not in columns:
            print(f"'{x}' is not a valid attribute name. Please choose from the list.")
        else:
            break

    while True:
        y = input("Select target attribute: ").strip()
        if y not in columns:
            print(f"'{y}' is not a valid attribute name. Please choose from the list.")
        else:
            break

    return x, y


def get_correlation_method():
    methods = ['pearson', 'spearman', 'kendall']
    default = 'pearson'
    method = input(f"Choose correlation method ({', '.join(methods)}) [default={default}]: ").strip().lower()

    if method == '':
        return default
    elif method in methods:
        return method
    else:
        print(f"Invalid method. Using default: {default}")
        return default


def get_alpha(default=0.1):
    try:
        alpha_input = input(f"\nEnter alpha value for pruning (α ∈ ⟨0, 0.3⟩) [default={default}]: ").strip()
        if alpha_input == '':
            return default
        alpha = float(alpha_input)
        if not (0 <= alpha <= 0.3):
            raise ValueError("Alpha must be between 0 and 0.3")
        return alpha
    except ValueError as e:
        print(f"Invalid input: {e}. Using default value: {default}")
        return default


def get_path_finding_method():
    methods = {
        'greedy': 'greedy',
        'greedy+dfs': 'greedy+dfs',
        'dfs': 'dfs',
        'a*': 'a_star'
    }
    default = 'greedy'
    prompt = f"Choose path finding method ({', '.join(methods.keys())}) [default={default}]: "
    method = input(prompt).strip().lower().replace(' ', '')

    return methods.get(method, default)


def get_frac(default=0.3):
    print("Regression analysis will be performed.")
    print("Please specify fraction value used for LOESS smoothing (controls smoothing degree).")
    try:
        frac_input = input(f"Enter fraction value (e.g., 0.3 for 30%) [default={default}]: ").strip()
        if frac_input == '':
            return default
        frac = float(frac_input)
        if not (0 < frac < 1):
            raise ValueError("Fraction must be between 0 and 1 (exclusive).")
        return frac
    except ValueError as e:
        print(f"Invalid input: {e}. Using default value: {default}")
        return default


def get_max_depth(default=5):
    print("Please specify max depth for CART (controls decision tree complexity).")
    try:
        max_depth_input = input(f"Enter max depth for CART (positive integer) [default={default}]: ").strip()
        if max_depth_input == '':
            return default
        max_depth = int(max_depth_input)
        if max_depth <= 0:
            raise ValueError("Max depth must be a positive integer.")
        return max_depth
    except ValueError as e:
        print(f"Invalid input: {e}. Using default value: {default}")
        return default


def get_plot_palette():
    palette_options = {
        1: 'meh',
        2: 'pastel-red',
        3: 'pastel-orange',
        4: 'bright'
    }
    default_key = 1
    prompt = (
        f"Available color palettes for the prediction error graph:\n"
        f"  1  default\n"
        f"  2  pastel-red\n"
        f"  3  pastel-orange\n"
        f"  4  bright\n"
        f"Enter a number to select a color palette from the list above [default={default_key}]: "
    )

    try:
        user_input = input(prompt).strip()
        key = int(user_input) if user_input else default_key
    except ValueError:
        key = default_key

    return palette_options.get(key, palette_options[default_key])
