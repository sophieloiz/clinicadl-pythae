from clinicadl.utils.caps_dataset.data import CapsDatasetImage
from pythae.data.datasets import DatasetOutput

class PythaeCAPS(CapsDatasetImage):
    def __init__(
        self,
        caps_directory,
        data_file,
        preprocessing_dict,
        train_transformations,
        all_transformations,
    ):
        super().__init__(
            caps_directory,
            data_file,
            preprocessing_dict,
            train_transformations=train_transformations,
            label_presence=False,
            all_transformations=all_transformations,
        )

    def __getitem__(self, index):
        X = super().__getitem__(index)
        return DatasetOutput(
            data=X["image"],
            participant_id=X["participant_id"],
            session_id=X["session_id"],
            image_id=X["image_id"],
            image_path=X["image_path"],
        )
