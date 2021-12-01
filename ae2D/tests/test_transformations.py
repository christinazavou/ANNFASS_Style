import numpy as np
import pytest
import torch
import torchvision.transforms as transforms

from ae2D.utils.transformations import DataSetUtils, DataSetHelper, DatasetFromNumpy, ToTensorDict, \
    NormalizeDict, InverseNormalizeDict, InverseToTensorDict, ToOneChannelDict, interval_mapping, normalize

img1 = np.array([
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], ],
    [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], ],
    [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], ]
])
img2 = np.array([
    [[105, 105, 105, 105], [105, 105, 105, 105], [105, 105, 105, 105], [105, 105, 105, 105], ],
    [[200, 200, 200, 200], [200, 200, 200, 200], [200, 200, 200, 200], [200, 200, 200, 200], ],
    [[40, 40, 40, 40], [40, 40, 40, 40], [40, 40, 40, 40], [40, 40, 40, 40], ]
])

img3 = np.array([
    [[0, 0, 0, 0], [0, 0, 0, 0], ],
    [[2, 2, 2, 2], [2, 2, 2, 2], ],
    [[4, 4, 4, 4], [4, 4, 4, 4], ]
])
img4 = np.array([
    [[105, 106, 107, 108], [109, 101, 102, 103], ],
    [[200, 202, 203, 204], [205, 200, 200, 200], ],
    [[40, 41, 40, 42], [40, 44, 40, 40], ]
])

test_images = np.array([img1, img2])
test_labels = [np.array([4.]), np.array([2.])]
test_images_34 = np.array([img3, img4])


def test_get_mean_function():
    loader = DataSetHelper.get_dataloader_from_examples(test_images, test_labels)
    assert np.array_equal(DataSetUtils.get_mean_function(loader, np.mean), np.array([52.5, 101., 22.]))
    assert np.array_equal(DataSetUtils.get_mean_function(loader, np.std), np.array([0, 0., 0.]))


def test_dict_transforms_correct_order():
    with pytest.raises(AssertionError) as error:
        transform = transforms.Compose([
            ToTensorDict(),
            NormalizeDict((52.5, 101., 22.), (0.1, 0.1, 0.1))
        ])
        test_data = DatasetFromNumpy(test_images, test_labels, transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=2, num_workers=0)
        for data in test_loader:
            pass
    assert "Not accepted input type." in str(error.value)


def test_normalize_dict_std():
    with pytest.raises(ValueError) as error:
        transform = transforms.Compose([
            ToTensorDict(),
            NormalizeDict((52.5, 101., 22.), (0., 0., 0.))
        ])
        test_data = DatasetFromNumpy(test_images, test_labels, transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=2, num_workers=0)
        for data in test_loader:
            pass
    assert "std evaluated to zero, leading to division by zero." in str(error.value)


def test_correct_normalization():
    transform = transforms.Compose([
        NormalizeDict((52.5, 101., 22.), (0.1, 0.1, 0.1)),
        ToTensorDict(),
    ])
    test_data = DatasetFromNumpy(test_images, test_labels, transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=2, num_workers=0)
    for data in test_loader:
        min_value = np.min(data['image'].numpy())
        max_value = np.max(data['image'].numpy())
        assert np.allclose(-max_value, min_value, rtol=5)

    test_loader = DataSetHelper.get_dataloader_from_examples(test_images_34, test_labels)
    mean = DataSetUtils.get_mean_function(test_loader, np.mean)
    std = DataSetUtils.get_mean_function(test_loader, np.std)
    transform = transforms.Compose([
        NormalizeDict(mean, std),
        ToTensorDict(),
    ])
    test_data = DatasetFromNumpy(test_images_34, test_labels, transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=2, num_workers=0)
    for data in test_loader:
        min_value = np.min(data['image'].numpy())
        max_value = np.max(data['image'].numpy())
        assert np.allclose(-max_value, min_value, rtol=5)


def test_correct_inverse_normalization():
    transform = transforms.Compose([
        NormalizeDict((52.5, 101., 22.), (0.1, 0.1, 0.1)),
        InverseNormalizeDict((52.5, 101., 22.), (0.1, 0.1, 0.1)),
    ])
    test_data = DatasetFromNumpy(test_images, test_labels, transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=2, num_workers=0)
    for data in test_loader:
        assert np.allclose(data['image'], test_images)
        assert np.array_equal(data['labels'], test_labels)
        break


def test_inverse_to_tensor_dict():
    transform = transforms.Compose([
        ToTensorDict(),
    ])
    inverse_transform = transforms.Compose([
        InverseToTensorDict(),
    ])
    test_data = DatasetFromNumpy(test_images, test_labels, transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=2, num_workers=0)
    for data in test_loader:
        data = inverse_transform(data)
        assert np.array_equal(data['image'], test_images)
        assert np.array_equal(data['labels'], test_labels)
        break


def test_to_one_channel_dict():
    test_data1 = DatasetFromNumpy(test_images.transpose((0, 2, 3, 1)), test_labels,
                                  transform=transforms.Compose([ToTensorDict(), ]))
    test_data2 = DatasetFromNumpy(test_images.transpose((0, 2, 3, 1)), test_labels,
                                  transform=transforms.Compose([ToTensorDict(), ToOneChannelDict()]))
    test_data3 = DatasetFromNumpy(test_images.transpose((0, 2, 3, 1)).astype(np.float), test_labels,
                                  transform=transforms.Compose([ToTensorDict(), ToOneChannelDict()]))
    test_loader1 = torch.utils.data.DataLoader(test_data1, batch_size=2, num_workers=0)
    test_loader2 = torch.utils.data.DataLoader(test_data2, batch_size=2, num_workers=0)
    test_loader3 = torch.utils.data.DataLoader(test_data3, batch_size=2, num_workers=0)
    for data in test_loader1:
        assert data['image'].size() == torch.Size([2, 3, 4, 4])
        assert data['image'].dtype in [torch.uint8, torch.int32, torch.int64]
        break
    for data in test_loader2:
        assert data['image'].size() == torch.Size([2, 1, 4, 4])
        assert data['image'].dtype in [torch.uint8, torch.int32, torch.int64]
        break
    for data in test_loader3:
        assert data['image'].size() == torch.Size([2, 1, 4, 4])
        assert data['image'].dtype in [torch.float, torch.float32, torch.float64]
        break


def test_interval_mapping():
    a = np.array([1, 127.5, 255])
    res = interval_mapping(a, 0, 255, 0, 1)
    assert np.array_equal(res, np.array([1/255., 0.5, 1]))

    a = np.array([0, 0.5, 1])
    res = interval_mapping(a, 0, 1, -1, 1)
    assert np.array_equal(res, np.array([-1, 0, 1.]))

    a = np.array([0, 0.5, 1])
    res = interval_mapping(a, 0, 1, 0, 255)
    assert np.array_equal(res, np.array([0, 127.5, 255]))


def test_normalize():
    a = np.array([1, 127.5, 255])
    res = normalize(a, 127.5, 1)
    print(res)
    res = normalize(a, 255, 1)
    print(res)
