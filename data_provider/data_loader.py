import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from utils.normalizer import UnitTransformer, UnitGaussianNormalizer


class DrivAerML(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.data = os.listdir(args.data_path)
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'.")

    def get_loader(self, full_mesh=False):
        data_list_x = []
        data_list_y = []
        data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                     28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 54, 55,
                     56, 57, 58, 59, 61, 62, 63, 65, 66, 68, 69, 74, 75, 76, 77, 78, 85, 86, 87, 88, 89, 90, 91, 92, 93,
                     101, 102, 103, 104, 44, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 145, 146, 149, 150, 151,
                     152, 153, 154, 155, 156, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                     122, 123, 124, 125, 127, 128, 129, 130, 131, 132]
        rng = np.random.default_rng(seed=0)
        for i in data_list:
            x = np.load(os.path.join(self.data_path, f"x_{i}.npy"))
            if full_mesh:
                y = np.load(os.path.join(self.data_path, f"y_{i}.npy"))
            else:
                x = np.load(os.path.join(self.data_path, f"x_{i}.npy"))
                idx = rng.choice(x.shape[0], size=70000, replace=False)
                x = x[idx, :]
                y = np.load(os.path.join(self.data_path, f"y_{i}.npy"))[idx, :]
            data_list_x.append(x)
            data_list_y.append(y)
        data_list_x = torch.tensor(np.array(data_list_x), dtype=torch.float)
        data_list_y = torch.tensor(np.array(data_list_y), dtype=torch.float)

        train_y = data_list_y[:self.ntrain, :, :]
        train_pos = data_list_x[:self.ntrain, :, :]
        train_cond = torch.ones_like(data_list_x[:self.ntrain, :1, :1])
        test_y = data_list_y[-self.ntest:, :, :]
        test_pos = data_list_x[-self.ntest:, :, :]
        test_cond = torch.ones_like(data_list_x[-self.ntest:, :1, :1])

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == 'UnitTransformer':
                self.y_normalizer = UnitTransformer(train_y)
            elif self.norm_type == 'UnitGaussianNormalizer':
                self.y_normalizer = UnitGaussianNormalizer(train_y)

            train_y = self.y_normalizer.encode(train_y)
            self.y_normalizer.cuda()

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_pos, train_pos, train_cond, train_y),
            batch_size=self.batch_size,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_pos, test_pos, test_cond, test_y),
            batch_size=self.batch_size,
            shuffle=False)
        print("Dataloading is over.")
        return train_loader, test_loader, [train_y.shape[1]]


class NASA(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.data = os.listdir(args.data_path)
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'.")

    def get_loader(self, full_mesh=False):
        train_pos = []
        train_y = []
        train_cond = []
        rng = np.random.default_rng(seed=0)
        for i in range(self.ntrain):
            cond = np.load(os.path.join(self.data_path, f"./train/cond_{i + 1}.npy"))
            x = np.load(os.path.join(self.data_path, f"./train/x_{i + 1}.npy"))
            if full_mesh:
                y = np.load(os.path.join(self.data_path, f"./train/y_{i + 1}.npy"))
            else:
                idx = rng.choice(x.shape[0], size=50000, replace=False)
                x = x[idx, :]
                y = np.load(os.path.join(self.data_path, f"./train/y_{i + 1}.npy"))[idx]
            train_pos.append(x)
            train_y.append(y)  # 105 50000
            train_cond.append(cond)  # 105 2
        train_pos = torch.tensor(np.array(train_pos), dtype=torch.float)
        train_cond = torch.tensor(np.array(train_cond), dtype=torch.float)[:, None, :]
        train_y = torch.tensor(np.array(train_y), dtype=torch.float)[:, :, None]

        test_pos = []
        test_y = []
        test_cond = []
        rng = np.random.default_rng(seed=0)
        for i in range(self.ntest):
            cond = np.load(os.path.join(self.data_path, f"./test/cond_{i + 1}.npy"))
            x = np.load(os.path.join(self.data_path, f"./test/x_{i + 1}.npy"))
            if full_mesh:
                y = np.load(os.path.join(self.data_path, f"./test/y_{i + 1}.npy"))
            else:
                idx = rng.choice(x.shape[0], size=50000, replace=False)
                x = x[idx, :]
                y = np.load(os.path.join(self.data_path, f"./test/y_{i + 1}.npy"))[idx]
            test_pos.append(x)
            test_y.append(y)  # 105 50000
            test_cond.append(cond)  # 105 2
        test_pos = torch.tensor(np.array(test_pos), dtype=torch.float)
        test_cond = torch.tensor(np.array(test_cond), dtype=torch.float)[:, None, :]
        test_y = torch.tensor(np.array(test_y), dtype=torch.float)[:, :, None]
        print(train_pos.shape, train_y.shape, test_pos.shape, test_y.shape)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == 'UnitTransformer':
                self.y_normalizer = UnitTransformer(train_y)
            elif self.norm_type == 'UnitGaussianNormalizer':
                self.y_normalizer = UnitGaussianNormalizer(train_y)

            train_y = self.y_normalizer.encode(train_y)
            self.y_normalizer.cuda()

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_pos, train_pos, train_cond, train_y),
            batch_size=self.batch_size,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_pos, test_pos, test_cond, test_y),
                                                  batch_size=self.batch_size,
                                                  shuffle=False)
        print("Dataloading is over.")
        return train_loader, test_loader, [train_y.shape[1]]


class AirCraft(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.data = os.listdir(args.data_path)
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type

        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'.")

    def get_loader(self, full_mesh=False):
        print("loading aircraft...")
        data_x = []
        data_y = []
        data_cond = []
        rng = np.random.default_rng(seed=0)
        for i in range(150):
            if full_mesh:
                x = np.load(os.path.join(self.data_path, f"x_{i}.npy"))
                y = np.load(os.path.join(self.data_path, f"y_{i}.npy"))[:, :]
            else:
                x = np.load(os.path.join(self.data_path, f"x_{i}.npy"))
                idx = rng.choice(x.shape[0], size=30000, replace=False)
                x = x[idx, :]
                y = np.load(os.path.join(self.data_path, f"y_{i}.npy"))[idx, :]
            cond = np.load(os.path.join(self.data_path, f"cond_{i}.npy"))
            data_x.append(x)
            data_y.append(y)
            data_cond.append(cond)
        print("finished aircraft...")
        train_pos = torch.tensor(np.array(data_x[:self.ntrain]), dtype=torch.float)
        train_y = torch.tensor(np.array(data_y[:self.ntrain]), dtype=torch.float)
        train_cond = torch.tensor(np.array(data_cond[:self.ntrain]), dtype=torch.float)[:, None, :]

        test_pos = torch.tensor(np.array(data_x[-self.ntest:]), dtype=torch.float)
        test_y = torch.tensor(np.array(data_y[-self.ntest:]), dtype=torch.float)
        test_cond = torch.tensor(np.array(data_cond[-self.ntest:]), dtype=torch.float)[:, None, :]

        print(train_pos.shape, train_y.shape, train_cond.shape, test_pos.shape, test_y.shape, test_cond.shape)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == 'UnitTransformer':
                self.y_normalizer = UnitTransformer(train_y)
            elif self.norm_type == 'UnitGaussianNormalizer':
                self.y_normalizer = UnitGaussianNormalizer(train_y)

            train_y = self.y_normalizer.encode(train_y)
            self.y_normalizer.cuda()

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_pos, train_pos, train_cond, train_y),
            batch_size=self.batch_size,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_pos, test_pos, test_cond, test_y),
                                                  batch_size=self.batch_size,
                                                  shuffle=False)
        print("Dataloading is over.")
        return train_loader, test_loader, [train_y.shape[1]]


class DTCHull(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.data = os.listdir(args.data_path)
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type

        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'.")

    def get_loader(self, full_mesh=False):
        print("loading hull dataset...")
        data_x = []
        data_y = []
        data_cond = []
        rng = np.random.default_rng(seed=0)
        for i in range(130):
            if i in [102, 104, 107, 108, 110, 114, 122, 123, 127, 129]:  # avoid broken files
                continue
            x = np.load(os.path.join(self.data_path, f"x_{i + 1}.npy"))
            if full_mesh:
                y = np.load(os.path.join(self.data_path, f"y_{i + 1}.npy"))
            else:
                x = np.load(os.path.join(self.data_path, f"x_{i + 1}.npy"))
                idx = rng.choice(x.shape[0], size=80000, replace=False)
                x = x[idx, :]
                y = np.load(os.path.join(self.data_path, f"y_{i + 1}.npy"))[idx, :]
            if x.shape[0] > 200000:
                x = x[-229000:, :]
                y = y[-229000:, :]
            cond = np.load(os.path.join(self.data_path, f"cond_{i + 1}.npy"))
            data_x.append(x)
            data_y.append(y)
            data_cond.append(cond)
        print("finished hull...")
        train_pos = torch.tensor(np.array(data_x[:self.ntrain]), dtype=torch.float)
        train_y = torch.tensor(np.array(data_y[:self.ntrain]), dtype=torch.float)
        train_cond = torch.tensor(np.array(data_cond[:self.ntrain]), dtype=torch.float)[:, None]
        test_pos = torch.tensor(np.array(data_x[-self.ntest:]), dtype=torch.float)
        test_y = torch.tensor(np.array(data_y[-self.ntest:]), dtype=torch.float)
        test_cond = torch.tensor(np.array(data_cond[-self.ntest:]), dtype=torch.float)[:, None]

        print(train_pos.shape, train_y.shape, train_cond.shape, test_pos.shape, test_y.shape, test_cond.shape)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == 'UnitTransformer':
                self.y_normalizer = UnitTransformer(train_y)
            elif self.norm_type == 'UnitGaussianNormalizer':
                self.y_normalizer = UnitGaussianNormalizer(train_y)

            train_y = self.y_normalizer.encode(train_y)
            self.y_normalizer.cuda()

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_pos, train_pos, train_cond, train_y),
            batch_size=self.batch_size,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_pos, test_pos, test_cond, test_y),
                                                  batch_size=self.batch_size,
                                                  shuffle=False)
        print("Dataloading is over.")
        return train_loader, test_loader, [train_y.shape[1]]


class Car_Crash(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.data = os.listdir(args.data_path)
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type

        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'.")

    def get_loader(self, full_mesh=False):
        print("loading crash dataset...")
        data_x = []
        data_y = []
        data_cond = []
        rng = np.random.default_rng(seed=0)
        for i in range(130):
            x = np.load(os.path.join(self.data_path, f"x_{i}.npy"))
            if full_mesh:
                x = np.concatenate((x[:, :3], np.zeros((x.shape[0], 1)), x[:, -3:]), axis=-1)
                y = np.load(os.path.join(self.data_path, f"y_{i}.npy"))[:, :]
            else:
                idx = rng.choice(x.shape[0], size=100000, replace=False)
                x = np.concatenate((x[idx, :3], np.zeros((idx.shape[0], 1)), x[idx, -3:]), axis=-1)
                y = np.load(os.path.join(self.data_path, f"y_{i}.npy"))[idx, :]
            cond = np.load(os.path.join(self.data_path, f"cond_{i}.npy"))
            data_x.append(x)
            data_y.append(y)
            data_cond.append(cond)
        print("finished crash...")
        train_pos = torch.tensor(np.array(data_x[:self.ntrain]), dtype=torch.float)
        train_y = torch.tensor(np.array(data_y[:self.ntrain]), dtype=torch.float)
        train_cond = torch.tensor(np.array(data_cond[:self.ntrain]), dtype=torch.float)[:, None]

        test_pos = torch.tensor(np.array(data_x[-self.ntest:]), dtype=torch.float)
        test_y = torch.tensor(np.array(data_y[-self.ntest:]), dtype=torch.float)
        test_cond = torch.tensor(np.array(data_cond[-self.ntest:]), dtype=torch.float)[:, None]

        print(train_pos.shape, train_y.shape, train_cond.shape, test_pos.shape, test_y.shape, test_cond.shape)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == 'UnitTransformer':
                self.y_normalizer = UnitTransformer(train_y)
            elif self.norm_type == 'UnitGaussianNormalizer':
                self.y_normalizer = UnitGaussianNormalizer(train_y)

            train_y = self.y_normalizer.encode(train_y)
            self.y_normalizer.cuda()
            print(self.y_normalizer.mean)
            print(self.y_normalizer.std)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_pos, train_pos, train_cond, train_y),
            batch_size=self.batch_size,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_pos, test_pos, test_cond, test_y),
            batch_size=self.batch_size,
            shuffle=False)
        print("Dataloading is over.")
        return train_loader, test_loader, [train_y.shape[1]]
