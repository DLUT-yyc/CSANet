from .cityscapes import CitySegmentationTest, CitySegmentationTrain, CitySegmentationVal

datasets = {
	'cityscapes_train_fine': CitySegmentationTrain,
        # 'cityscapes_train_coarse':CitySegmentationTrainCoarse,
        'cityscapes_val':CitySegmentationVal,
	'cityscapes_test': CitySegmentationTest,
	# 'cityscapes_train_w_path': CitySegmentationTrainWpath,
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
