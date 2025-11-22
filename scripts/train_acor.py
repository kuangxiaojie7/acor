from pathlib import Path

from acor.runners import ACORTrainer
from acor.utils.config import build_argparser, load_config, parse_unknown_args


def main() -> None:
    parser = build_argparser("Train ACOR policy on PettingZoo environments.")
    args, unknown = parser.parse_known_args()
    overrides = parse_unknown_args(unknown)
    config = load_config(args.config, overrides)

    output_dir = args.output_dir or config["experiment"].get("output_dir")
    run_name = args.run_name or config["experiment"].get("run_name")
    if output_dir and run_name:
        output_dir = Path(output_dir) / run_name

    trainer = ACORTrainer(config=config, output_dir=str(output_dir) if output_dir else None)
    trainer.train()


if __name__ == "__main__":
    main()
