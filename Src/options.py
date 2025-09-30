# === args.py ===
import argparse
import numpy as np

def get_args():
    p = argparse.ArgumentParser("Federated TinyLlama config (underscore keys)")

    # Fed settings
    p.add_argument("--num_clients",      type=int,   default=10)
    p.add_argument("--run_serial",       type=parse_bool_tf, default=False, help="run clients serially")
    p.add_argument("--gpu_nums",         type=int,   default=2)
    p.add_argument("--times",            type=int,   default=10)
    p.add_argument("--noniid",           type=float,   default=0.3)

    #add noise to dataset
    p.add_argument("--noisy_set",        type=parse_bool_tf, default=False, help="help = enable noisy dataset split, True/False")
    p.add_argument("--noisy_proportion", type=float, default=0.5)
    p.add_argument("--noise_data_rate",  type=float, default=0.8)
    p.add_argument("--noise_mean",       type=float, default=1.0)
    p.add_argument("--noise_std",        type=float, default=2.0)

    #add noise to LoRA
    p.add_argument("--add_lora_noise",     type=parse_bool_tf, default=True,  help="add Gaussian noise to LoRA A/B")
    p.add_argument("--load_artifacts",     type=parse_bool_tf, default=True, help="load cached artifacts if available")
    p.add_argument("--weight_mode",     type=str, default='pca', help="weights allocation, opt: avg, rpca, pca")
    p.add_argument("--tau",             type=float, default=0.8, help="incentive strength")

    # noise scale
    p.add_argument("--noise_seed", type=int,   default=2025)
    p.add_argument("--sigma_min",  type=float, default=0.00)
    p.add_argument("--sigma_max",  type=float, default=0.1)
    p.add_argument("--gamma_seed", type=int,   default=2025)
    p.add_argument("--gamma_min",  type=float, default=1, help="it actually denotes 1/gamma = beta/alpha, the smaller gamma, the more the prefernece")
    p.add_argument("--gamma_max",  type=float, default=10)

    args = p.parse_args()

    if not (0.0 <= args.noisy_proportion <= 1.0):
        p.error("--noisy_proportion must be in [0,1]")
    if not (0.0 <= args.noise_data_rate <= 1.0):
        p.error("--noise_data_rate must be in [0,1]")
    if args.num_clients <= 0:
        p.error("--num_clients must be > 0")
    if args.sigma_max < args.sigma_min:
        p.error("--sigma_max must be >= --sigma_min")

    return args

def parse_bool_tf(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v == "True":
            return True
        if v == "False":
            return False
    raise argparse.ArgumentTypeError("expected True or False (case-sensitive), or a Python bool")

