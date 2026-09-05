# Adaptation data contract

No dataset is distributed. Local historical image folders are excluded from Git and are not automatically rewritten by these scripts.

Supply source data with semantic class names:

```text
data/tip_adapter/
├── train/
│   ├── Dog/
│   ├── Duck/
│   ├── Erhu/
│   ├── Piano/
│   └── Porcelain/
└── val/
    ├── Dog/
    ├── Duck/
    ├── Erhu/
    ├── Piano/
    ├── Porcelain/
    └── OtherCategory/
```

Every split must contain images of the selected target and at least one non-target category. Training requires at least `shots` images in each binary class. A Dog run retains Dog as positive and merges other source classes into Others. Additional confusable categories may be included. Folder names illustrate the format, not a supplied dataset.

`preprocessing.py --keep_classes Dog` creates:

```text
.local/tip_adapter/datasets/images/
├── .retrieval-generated.json
├── train/
│   ├── 0/Dog/       # Target label 0
│   └── 1/...        # Other source classes; label 1
└── val/
    ├── 0/Dog/
    └── 1/...
```

The metadata marker identifies this tool's generated output. Replacing it requires explicit overwrite; an arbitrary existing directory cannot be cleared. Source files are copied and never deleted. Preparation checks directory layout and image extensions, not content quality or deduplication.

Paths in `imagenet.yaml` are relative to the YAML directory. `target_dir` must equal `root_path/datasets/images`. Set `source_dir` to your complete source dataset. The existing private `tip_adapter_ft/datasets/val` folder is insufficient for this contract: it contains only background categories. No repair or re-splitting of private data is performed by publication cleanup.

The historical `val` split is also used as `test` by the original loader. This documentation does not claim an additional held-out split.
