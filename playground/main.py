import data
import predict
import testing
import compute


def main():
    """Main entry point."""
    #testing.torch_testing()
    #testing.torch_testing_regression()

    #data.save_data_interoperable()
    #data.save_data()

    compute.similarity()
    compute.dbscan()
    compute.k_span()


if __name__ == "__main__":
    main()


