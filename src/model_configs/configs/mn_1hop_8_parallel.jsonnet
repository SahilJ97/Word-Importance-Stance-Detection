{
    "dataset_reader": {
        "type": "vast_reader",
        return_separate: true
    },
    "train_data_path": "data/VAST_romanian/vast_train.csv",
    "validation_data_path": "data/VAST_romanian/vast_dev.csv",
    "model": {
        "type": "memory_network",
        "num_hops": 1,
        "text_embedding_size": 768,
        "hidden_layer_size": 200,
        "init_topic_knowledge_file": ../initial_topic_knowledge/from_top_doc/8.pt,
        "knowledge_transfer_scheme": parallel,
        "pretrained_model": "bert-base-multilingual-cased"
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
    }
}
