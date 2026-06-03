import argparse


def parse_token_args():
    parser = argparse.ArgumentParser(
        description="TRELLIS token pruning and sampler acceleration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode_group = parser.add_argument_group("Mode Control & Steps")
    mode_group.add_argument("--use_token", action="store_true", help="Token pruning acceleration.")
    mode_group.add_argument("--euler_steps", type=int, default=25, help="Euler sampling steps.")

    io_group = parser.add_argument_group("TRELLIS I/O & Internal Options")
    io_group.add_argument("--seed", type=int, default=42)
    io_group.add_argument("--resolution", type=int, default=16)

    args, _ = parser.parse_known_args()
    # effective_steps is overwritten per-run by the sampler with the actual step
    # count; the parser default only matters before the first sample() call.
    args.effective_steps = args.euler_steps
    args.use_token = True

    return args
