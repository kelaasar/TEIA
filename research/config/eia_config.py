"""Simple base config to follow."""
import argparse

def parse_argument():
    '''
    Provide basic configuration for model training.
    '''
    # Basic settings
    parser = argparse.ArgumentParser(description='Base config.')
    parser.add_argument('--exp_name', type=str, help='Experiment name.')
    parser.add_argument('--topk', type=int, nargs='+', default=[5, 10, 15])
    parser.add_argument('--dataset', type=str, default='qnli',
                        help='Name of dataset')
    parser.add_argument('--external_dataset', type=str, default='personachat',
                        help='Name of external dataset')
    parser.add_argument('--external_encoder', type=str, default='st5',
                        help='Name of external encoder for external data')
    parser.add_argument('--blackbox_encoder', type=str, default='sbert',
                        help='Name of blackbox encoder')
    parser.add_argument('--testing', action='store_true',
                        help='Set this flag to test the model in mini batch.')

    # Training parameters
    parser.add_argument('--train_ratio', type=float,
                        default=1.0, help='Ratio of training data.')
    parser.add_argument('--num_epochs', type=int,
                        default=10, help='Training epoches.')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='Batch_size #.')

    return parser.parse_args()
