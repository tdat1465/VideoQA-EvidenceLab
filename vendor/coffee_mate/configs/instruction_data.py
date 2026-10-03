import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy

anno_root_it = "your_annotation_path"

# ============== pretraining datasets=================
available_corpus = dict(
    reasoning_next_qa=[  # TODO
        f"/.../nextqa_train.json",
        "/.../nextqa/video/",
        "video"
    ],
    test_next_qa=[  # TODO
        f"/.../nextqa_val.json",
        "/.../nextqa/video/",
        "video"
    ],
    reasoning_star=[  # TODO
        f"/.../star_train.json",
        "/.../STAR/Charades_v1_480/",
        "video"
    ],
    test_star=[  # TODO
        f"/.../star_val.json",
        "/.../STAR/Charades_v1_480/",
        "video"
    ],
    reasoning_vlep=[  # TODO
        f"/.../vlep_train.json",
        "/.../vlep/vlep_frames/",
        "video"
    ],
    test_vlep=[  # TODO
        f"/.../vlep_val.json",
        "/.../vlep/vlep_frames/",
        "video"
    ],
    reasoning_tvqa=[  # TODO
        f"/.../tvqa_train.json",
        "/.../TVQA/tvqa_all_frames/",
        "video"
    ],
    test_tvqa=[  # TODO
        f"/.../tvqa_val.json",
        "/.../TVQA/tvqa_all_frames/",
        "video"
    ],
)


available_corpus["videochat2_instruction"] = [
    available_corpus["reasoning_next_qa"],
    available_corpus["test_next_qa"],
    available_corpus["reasoning_star"],
    available_corpus["test_star"],
    available_corpus["reasoning_vlep"],
    available_corpus["test_vlep"],
    available_corpus["reasoning_tvqa"],
    available_corpus["test_tvqa"],
]
