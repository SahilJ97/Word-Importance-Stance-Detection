import os
import glob

baseline_mbert = """{
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
    }
}
"""

memory_network_template = """{{
    "dataset_reader": {{
        "type": "vast_reader",
        return_separate: true
    }},
    "train_data_path": "data/VAST_romanian/vast_train.csv",
    "validation_data_path": "data/VAST_romanian/vast_dev.csv",
    "model": {{
        "type": "memory_network",
        "num_hops": {},
        "text_embedding_size": 768,
        "hidden_layer_size": 200,
        "init_topic_knowledge_file": {},
        "knowledge_transfer_scheme": {},
        "pretrained_model": "bert-base-multilingual-cased"
    }},
    "data_loader": {{
        batch_size: 64,
        shuffle: true
    }},
    "trainer": {{
        "optimizer": {{
            "type": "adam",
            "lr": 1e-3
        }},
        "num_epochs": 30,
        "grad_norm": 1.0,
        "patience": 10,
    }}
}}
"""

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    # Clean directory
    for j_file in glob.glob("configs/*.jsonnet"):
        os.remove(j_file)

    # Generate baseline model config
    with open("configs/baseline_mbert.jsonnet", "w+") as f:
        f.write(baseline_mbert)

    # Generate model configs for memory network
    for num_hops in range(1, 4):
        for init_topic_knowledge_file in glob.glob("../initial_topic_knowledge/from_top_doc/*.pt"):
            print(init_topic_knowledge_file)
            for knowledge_transfer_scheme in ["projection", "parallel"]:
                config = memory_network_template.format(
                    num_hops, init_topic_knowledge_file, knowledge_transfer_scheme
                )
                config_name = "configs/mn_{}hop_{}_{}.jsonnet".format(
                    num_hops,
                    init_topic_knowledge_file.split('.')[-2].split("/")[-1],
                    knowledge_transfer_scheme
                )
                with open(config_name, "w+") as f:
                    f.write(config)
