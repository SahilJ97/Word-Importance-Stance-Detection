{
    "dataset_reader": {
        "type": "vast_reader"
    },
    "train_data_path": "data/VAST_romanian/vast_train.csv",
    "validation_data_path": "data/VAST_romanian/vast_dev.csv",
    "model": {
        "type": "baseline_mbert"
    },
    "data_loader": {
        batch_size: 64,
        shuffle: true
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 1e-3
        },
        "num_epochs": 30,
        "grad_norm": 1.0,
        "patience": 10,
        "cuda_device": 2
    }
}
