import argparse
from src.train import main as train_main

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")

    args = parser.parse_args()

    if args.mode == "train":
        train_main()

if __name__ == "__main__":
    run()
