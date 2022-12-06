import argparse
from tensorflow import keras

def main(args):
    # example features for testing
    features = [
        [0.846154, 0.405405, 0.714286]
    ]
    # expected value is 0.379982 using dummy model (dummy.ckpt)
    error_estimate_model = keras.models.load_model(args.model)
    print(error_estimate_model.predict(features))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-model',
                        '--model',
                        required=True,
                        help="path to training data")

    # add more arguments here 
    # .. 

    args = parser.parse_args()
    main(args)
