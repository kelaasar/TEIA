"""Simple base config to follow."""
import argparse


def parse_argument():
    '''
    Provide basic configuration for model training.
    '''
    # Basic settings
    parser = argparse.ArgumentParser(description='Base config.')
    parser.add_argument('--project_name', type=str, required=True, help='Project name.')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name.')
    parser.add_argument('--model_dir', type=str,
                        default='microsoft/DialoGPT-small', help='Dir of your attack model')
    parser.add_argument('--dataset', type=str, default='qnli', choices=['qnli', 'personachat', 'IMDB', 'agnews', 'MIMC'],
                        help='Name of dataset')
    parser.add_argument('--external_dataset', type=str, default='personachat', choices=['qnli', 'personachat', 'IMDB', 'agnews'],
                        help='Name of external dataset')
    parser.add_argument('--surrogate_encoder', type=str, default='gtr-base',
                        help='Name of surrogate encoder')
    parser.add_argument('--external_encoder', type=str, default='st5',
                        help='Name of external encoder for external data')
    parser.add_argument('--blackbox_encoder', type=str, default='sbert', choices=['sbert', 'st5', 'openai'],
                        help='Name of blackbox encoder')
    parser.add_argument('--geia', action='store_true',
                        help='Set this flag to use GEIA otherwise use AdvGEIA')
    parser.add_argument('--testing', action='store_true',
                        help='Set this flag to test the model in mini batch.')

    # Training parameters
    parser.add_argument('--train_ratio', type=float,
                        default=0.1, help='Ratio of training data.')
    parser.add_argument('--training_size', type=int,
                        default=8000, help='Count of training data.')
    parser.add_argument('--external_size', type=int,
                        default=50000, help='Count of external data.')
    parser.add_argument('--surrogate_epoch', type=int,
                        default=10, help='Surrogate training epoches.')
    parser.add_argument('--num_epochs', type=int,
                        default=24, help='Training epoches.')
    parser.add_argument('--eval_per_epochs', type=int,
                        default=2, help='Evaluate per # epoches.')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='Batch_size #.')
    parser.add_argument('--mapping_lambda', type=float,
                        default=1.0, help='Lambda for mapping loss.')
    parser.add_argument('--pivot_lambda', type=float,
                        default=1.0, help='Lambda for pivot loss.')

    # Decoding parameters
    parser.add_argument('--model_epoch', type=int,
                        default=-1, help='Checkpoint epoch of attack model.')
    parser.add_argument('--decode', type=str, default='sampling',
                        help='Name of decoding methods: beam/sampling')

    # Easy data augmentation
    parser.add_argument('--multiple', type=int,
                        default=5, help='multiple of producing new data')
    parser.add_argument('--option', type=str,
                        default='llm', help='option for data augmentation', choices=['swap', 'insert', 'replace', 'delete', 'llm', 'None'])

    return parser.parse_args()
