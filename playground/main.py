import data
import predict
import testing


def main():
    """Main entry point."""
    #testing.torch_testing()
    #testing.torch_testing_regression()

    data.save_data_interoperable()

    #predict.predict_data()


if __name__ == "__main__":
    main()


