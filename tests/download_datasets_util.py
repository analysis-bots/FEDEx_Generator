"""
A short script for downloading all of the datasets in the default datasets file.
"""

from test_utils import load_datasets

if __name__ == '__main__':
    datasets = [
        {
            'first_dataset': '',
            'dataset_name': 'adult',
            'second_dataset': '',
            'second_dataset_name': 'adult'
        },
        {
            'first_dataset': '',
            'dataset_name': 'spotify_all',
            'second_dataset': '',
            'second_dataset_name': 'spotify_all'
        },
        {
            'first_dataset': '',
            'dataset_name': 'bank_churners_user_study',
            'second_dataset': '',
            'second_dataset_name': 'bank_churners_user_study'
        },
        {
            'first_dataset': '',
            'dataset_name': 'houses',
            'second_dataset': '',
            'second_dataset_name': 'houses'
        }
    ]

    load_datasets(datasets)
