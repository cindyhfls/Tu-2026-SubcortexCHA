import psutil,os,json,sys
from datetime import datetime

def print_memory_usage(message=""):
    mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"{datetime.now().strftime('%H:%M:%S')} | {mem_gb:.2f} GB {message}")

def load_args_from_json(json_file):
    """Load arguments from JSON file"""
    with open(json_file, 'r') as f:
        config = json.load(f)

    # Convert back to proper types where needed
    # Handle special types that might be lost in JSON serialization
    if 'cv_alphas' in config and isinstance(config['cv_alphas'], list):
        config['cv_alphas'] = [float(x) for x in config['cv_alphas']]
    if 'run' in config and isinstance(config['run'], list):
        if isinstance(config['run'][0],float):
            config['run'] = [int(x) for x in config['run']]
    return config

def git_provenance(path):
    """Commit of the git repo holding `path`, and whether `path` differs from it.

    A dirty (modified or untracked) file is not described by the commit, so its
    SHA-256 is recorded too. Outside a git repo, commit is None.
    """
    import subprocess, hashlib
    path = os.path.abspath(path)
    def git(*a):
        return subprocess.run(["git", "-C", os.path.dirname(path), *a],
                              capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "--short", "HEAD") or None
    out = {"file": path, "commit": commit}
    if commit:
        out["dirty"] = bool(git("status", "--porcelain", "--", path))
        if out["dirty"] and os.path.isfile(path):
            with open(path, "rb") as f:
                out["sha256"] = hashlib.sha256(f.read()).hexdigest()
    return out

def save_args_to_json(args, output_dir=None):
    """Save all arguments to a JSON file"""
    if output_dir is None:
        output_dir = args.outdir if args.outdir else "."

    output_dir = os.path.join(output_dir,'config')
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"config_{args.stage}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert args to dictionary
    args_dict = vars(args).copy()
    args_dict["provenance"] = {"script": git_provenance(sys.argv[0]),
                               "utils": git_provenance(__file__)}

    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(args_dict, f, indent=2, default=str)

    print(f"Configuration saved to: {filepath}")
    return filepath

def merge_args(json_dict, cli_ns):
    # second argument can overwrite the first

    merged = json_dict.copy()
    cli_dict = vars(cli_ns)

    # args actually typed on CLI
    typed = set(
        a.lstrip('-').replace('-', '_')
        for a in sys.argv[1:]
        if a.startswith('-')
    )

    for key, value in cli_dict.items():
        if key in typed:   # only override if actually typed
            merged[key] = value
        else: # if it is not typed, replace it only if it does not exist already
            if key not in merged:
                merged[key] = value
    return merged
