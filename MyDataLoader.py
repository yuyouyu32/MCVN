import os

import numpy as np
import pandas as pd
import torch

from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import Tensor
import torch.utils.data as Data


class MyDataLoader:
    """This class defines the data loading methods.

    Attributes:
        data_path: The file path of data.
        img_path: The file path of images.
        targets: The targets' names of job.

    """

    def __init__(self, data_path: str, img_path: str, targets: List[str]) -> None:
        if os.path.isfile(data_path) and os.path.isfile(img_path):
            self.data_path = data_path
            self.img_path = img_path
        else:
            raise Exception(
                "Sorry, the path of data or path of images is illegal.")
        if len(targets) <= 0:
            raise Exception(
                "The names of targets can not be empty.")
        self.targets = targets

    def set_data_path(self, data_path: str) -> None:
        """Set new path of data.

        Arguments:
            data_path: The file path of data.

        """
        if os.path.isfile(data_path):
            self.save_path = data_path
        else:
            raise Exception("Sorry, the path of data is illegal.")

    def set_img_path(self, img_path: str) -> None:
        """Set new path of images.

        Arguments:
            img_path: The direction path of images.

        """
        if os.path.isfile(img_path):
            self.save_path = img_path
        else:
            raise Exception("Sorry, the path of saving is illegal.")

    def set_targets(self, targets: List[str]) -> None:
        """Set new names of targets.

        Arguments:
            targets: The targets' names of job.

        """
        if len(targets) <= 0:
            raise Exception(
                "The names of targets can not be empty.")
        self.targets = targets

    def get_dataset(self, normal_feature: bool = False, normal_target: bool = False,
                    valid_size: float = 0.25, test_size: float = 0.2,
                    normalized_method: str = 'MinMaxScaler') -> Tuple[Dict[str, Data.TensorDataset],
                                                                      Dict[str,
                                                                           List[Tensor]],
                                                                      Dict[str, List[Tensor]]]:
        """Get Train Valid Test TensorDataset.

        Arguments:
            normal_feature: if normalize features.
            normal_target: if normalize targets.
            valid_size: valid dataset size.
            test_size: test dataset size.
            normalized_method: MinMaxScale or StandardScaler

        Return:
            Tuple of Train Valid Test Tensor Dataset.

        """
        # Image channel.
        img_data = pd.read_csv(self.img_path).astype('double')
        img_data.fillna(0, inplace=True)
        img_data.replace(np.inf, 255, inplace=True)
        # Targets and original channel.
        original_data = pd.read_csv(self.data_path).astype('double')
        targets = original_data.loc[:, self.targets].values
        features = original_data.drop(columns=self.targets).values
        features_number = features.shape[1]

        # Nomrmalization
        if normal_feature:
            if normalized_method == 'MinMaxScaler':
                scaler = MinMaxScaler()
            elif normalized_method == 'StandardScaler':
                scaler = StandardScaler()
            else:
                raise Exception(
                    'Normalized method name should be MinMaxScaler or StandardScaler.')
            features = scaler.fit_transform(features)
        if normal_target:
            if normalized_method == 'MinMaxScaler':
                scaler_target = MinMaxScaler()
            elif normalized_method == 'StandardScaler':
                scaler_target = StandardScaler()
            else:
                raise Exception(
                    'Normalized method name should be MinMaxScaler or StandardScaler.')
            targets = scaler_target.fit_transform(targets)
        # Combine original and image channel.
        combine_data = np.concatenate((img_data, features), axis=1)
        # Get three dataset.
        Train, Valid, Test = {}, {}, {}
        for index, target_name in enumerate(self.targets):
            x_train_val, x_test, y_train_val, y_test = train_test_split(
                combine_data, targets[:, index], test_size=test_size, random_state=42)
            x_train, x_val, y_train, y_val = train_test_split(
                x_train_val, y_train_val, test_size=valid_size, random_state=42)
            x_train_pics = torch.from_numpy(x_train[:, :-features_number])
            x_val_pics = torch.from_numpy(x_val[:, :-features_number])
            x_test_pics = torch.from_numpy(x_test[:, :-features_number])
            x_train_other = torch.from_numpy(x_train[:, -features_number:])
            x_val_other = torch.from_numpy(x_val[:, -features_number:])
            x_test_other = torch.from_numpy(x_test[:, -features_number:])
            y_train = torch.from_numpy(y_train)
            y_val = torch.from_numpy(y_val)
            y_test = torch.from_numpy(y_test)
            Train[target_name] = Data.TensorDataset(
                x_train_pics, x_train_other, y_train)
            Valid[target_name] = [x_val_pics, x_val_other, y_val]
            Test[target_name] = [x_test_pics, x_test_other, y_test]
        return Train, Valid, Test


'''
# Test
if __name__ == '__main__':
    dataloader = MyDataLoader(data_path='./Data/NIMS/NIMS_Fatigue.csv', img_path='./Data/NIMS/Images.csv', targets=['Fatigue', 'Tensile', 'Fracture', 'Hardness'])
    Train, Valid, Test = dataloader.get_dataset(normal_feature=True)
'''
