from src.experiment import run_all_experiments
from src.plotting import generate_all_plots


def main():

    run_all_experiments()

    generate_all_plots()


if __name__ == "__main__":
    main()