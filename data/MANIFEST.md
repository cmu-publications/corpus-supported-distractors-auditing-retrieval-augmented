# Data manifest

The datasets this experiment reads, with a hash for every file so a copy can be checked against the one the results were measured on.

## Included here

| Dataset | Size | Files |
|---|---|---|
| `openbookqa` | 1.5 MB | 4 |
| `sciq` | 7.3 MB | 4 |

The loaders in `code/data.py` take a dataset name and a directory, so any dataset missing here is downloaded on first run.

```
306433de1af97874  data/openbookqa/allenai___openbookqa/additional/0.0.0/388097ea7776314e93a529163e0fea805b8a6454/dataset_info.json
82b62a5dc8800f08  data/openbookqa/allenai___openbookqa/additional/0.0.0/388097ea7776314e93a529163e0fea805b8a6454/openbookqa-test.arrow
dfb6603357646474  data/openbookqa/allenai___openbookqa/additional/0.0.0/388097ea7776314e93a529163e0fea805b8a6454/openbookqa-train.arrow
e1c26f0145c565a6  data/openbookqa/allenai___openbookqa/additional/0.0.0/388097ea7776314e93a529163e0fea805b8a6454/openbookqa-validation.arrow
fee9fba1e2cb2cdf  data/sciq/allenai___sciq/default/0.0.0/2c94ad3e1aafab77146f384e23536f97a4849815/dataset_info.json
955f96a8a6e7853a  data/sciq/allenai___sciq/default/0.0.0/2c94ad3e1aafab77146f384e23536f97a4849815/sciq-test.arrow
9b7f0d56d12d479e  data/sciq/allenai___sciq/default/0.0.0/2c94ad3e1aafab77146f384e23536f97a4849815/sciq-train.arrow
656cdf16f4061736  data/sciq/allenai___sciq/default/0.0.0/2c94ad3e1aafab77146f384e23536f97a4849815/sciq-validation.arrow
```
