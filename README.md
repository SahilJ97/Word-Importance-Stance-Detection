
## Example training commands (run from project root)

### BaselineMBERT:

    $ allennlp train src/BaselineMBERT.jsonnet -s ./baseline_mbert \
    --include-package src.classifiers --include-package allennlp.models --include-package src.vast_reader -f

### MemoryNetwork:

    $ allennlp train src/MemoryNetwork.jsonnet -s ./memory_network \
    --include-package src.classifiers --include-package allennlp.models --include-package src.vast_reader -f

## Example testing commands
### BaselineMBERT:
    $ allennlp evaluate baseline_mbert/model.tar.gz data/VAST_romanian/vast_test.csv \
    --include-package src.classifiers --include-package allennlp.models --include-package src.vast_reader