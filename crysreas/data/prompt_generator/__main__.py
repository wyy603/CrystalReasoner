import tyro

from .cli import Args, main

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
