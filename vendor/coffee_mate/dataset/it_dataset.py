import logging
import os
import json
import sqlite3
import random
from os.path import basename

import numpy as np
import datetime

from dataset.base_dataset import ImageVideoBaseDataset
from dataset.utils import load_anno
from dataset.video_utils import VIDEO_READER_FUNCS
from utils.distributed import is_main_process

logger = logging.getLogger(__name__)


class ITImgTrainDataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(
        self, ann_file, transform, 
        system="", role=("Human", "Assistant"),
        start_token="<Image>", end_token="</Image>",
        random_shuffle=True, # if True, shuffle the QA list
    ):
        super().__init__()

        if len(ann_file) == 3 and ann_file[2] == "video":
            self.media_type = "video"  
        else:
            self.media_type = "image"
        self.label_file, self.data_root = ann_file[:2]

        logger.info('Load json file')
        with open(self.label_file, 'r') as f:
            self.anno = json.load(f)
        self.num_examples = len(self.anno)
        self.transform = transform

        # prompt parameters
        if system:
            assert system[-1] == " ", "' ' should be add in the end of system, thus '###' will be tokenized into one token."
        # currently not support add start_token and end_token in the system, since the msg should be added properly
        self.begin_signal = "###"
        self.end_signal = " "
        self.start_token = start_token
        self.end_token = end_token
        self.system = system
        self.role = role
        self.random_shuffle = random_shuffle
        # instruction location and number
        logger.info(f"Random shuffle: {self.random_shuffle}")

    def get_anno(self, index):
        filename = self.anno[index][self.media_type]
        qa = self.anno[index]["QA"]
        question_type = self.anno[index]["type"] if "type" in self.anno[index] else None
        if "start" in self.anno[index] and "end" in self.anno[index]:
            anno = {
                "image": os.path.join(self.data_root, filename), "qa": qa, "question_type": question_type, "video_id": self.anno[index]["video"],
                "start": self.anno[index]["start"], "end": self.anno[index]["end"],
            }
        else:
            anno = {"image": os.path.join(self.data_root, filename), "qa": qa, "question_type": question_type, "video_id": self.anno[index]["video"], }
        return anno

    def __len__(self):
        return self.num_examples
    
    def process_qa(self, qa, msg=""):
        cur_instruction = ""
        # randomly shuffle qa for conversation
        if self.random_shuffle and len(qa) > 1:
            random.shuffle(qa)
        if "i" in qa[0].keys() and qa[0]["i"] != "":
            cur_instruction = qa[0]["i"] + self.end_signal

        conversation = self.system
        # add instruction as system message
        if cur_instruction:
            conversation += cur_instruction

        # rstrip() for the extra " " in msg
        conversation += (
            self.begin_signal + self.role[0] + ": " + 
            self.start_token + self.end_token + msg.rstrip() + self.end_signal
        )

        question_prompt = '\nOnly give the best option. '
        for sentence in qa:
            q = sentence["q"]
            a = sentence["a"]
            if "d" in sentence.keys() and sentence["d"] != "":
                conversation += (sentence["d"] + self.end_signal)

            if q != "":
                conversation += (self.begin_signal + self.role[0] + ": " + q + self.end_signal)
            else:
                # no question, often in caption dataset
                pass
            # generation = conversation + (self.begin_signal + self.role[1] + ": " + 'Answer: (')
            generation = conversation + (question_prompt + self.begin_signal + self.role[1] + ": " + 'Best option: (')
            answer = (a[9:])  # except for "Answer: ("
            conversation += (self.begin_signal + self.role[1] + ": " + a + self.end_signal)
        conversation += self.begin_signal
        if cur_instruction:
            cur_instruction += qa[0]["q"]

        return conversation, cur_instruction.strip(), qa[0]["q"], generation, answer

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            image, index = self.load_and_transform_media_data_image(index, ann["image"])
            conversation, instruction, _, _ = self.process_qa(ann["qa"])
            return image, conversation, instruction, index
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading image {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


class ITVidTrainDataset(ITImgTrainDataset):
    media_type = "video"

    def __init__(
        self, ann_file, transform,
        num_frames=4, video_reader_type="decord", sample_type="rand", num_tries=3,
        system="", role=("Human", "Assistant"),
        start_token="<Video>", end_token="</Video>",
        add_second_msg=True,
        random_shuffle=True,
    ):
        super().__init__(
            ann_file, transform, 
            system=system, role=role,
            start_token=start_token, end_token=end_token,
            random_shuffle=random_shuffle,
        )
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.add_second_msg = add_second_msg
        self.question_type_list, self.video_list = self.group(self.anno)
        logger.info(f"Use {video_reader_type} for data in {ann_file}")
        if add_second_msg:
            logger.info(f"Add second message: The video contains X frames sampled at T seconds.")

    def group(self, data):
        question_type_list = {}
        video_list = {}
        for sample in data:
            qa = sample["QA"][0]['q'] + sample["QA"][0]['a']
            type = sample["type"]
            video = sample["video"]
            if type == 'TP':
                type = 'TN'
            if type not in question_type_list:
                question_type_list[type] = {qa}
            else:
                question_type_list[type].add(qa)
            if video not in video_list:
                video_list[video] = {qa}
            else:
                video_list[video].add(qa)
        return question_type_list, video_list

    def __getitem__(self, index):
        ann = self.get_anno(index)
        msg = ""
        clip = None
        if "start" in ann and "end" in ann:
            clip = [ann["start"], ann["end"]]
        video, index, sec = self.load_and_transform_media_data_video(index, ann["image"], return_fps=True, clip=clip)
        sec = sec[::2]
        if self.add_second_msg:
            # " " should be added in the start and end
            msg = f" The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds. "
        conversation, instruction, question, _, _ = self.process_qa(ann["qa"], msg)

        question_type = ann["question_type"]
        video_id = ann["video_id"]
        if question_type == 'TP':
            question_type = 'TN'
        question_type_list = self.question_type_list[question_type]
        video_list = self.video_list[video_id]
        currnet_sample = ann["qa"][0]['q'] + ann["qa"][0]['a']
        neg_list = question_type_list - set(currnet_sample) - video_list
        neg_q = random.sample(list(neg_list), 9)
        neg_q.append(currnet_sample)
        random.shuffle(neg_q)
        qsn_id = neg_q.index(currnet_sample)

        return video, conversation, instruction, question, index, neg_q, qsn_id


class ITVidTestDataset(ITImgTrainDataset):
    media_type = "video"

    def __init__(
        self, ann_file, transform,
        num_frames=4, video_reader_type="decord", sample_type="rand", num_tries=3,
        system="", role=("Human", "Assistant"),
        start_token="<Video>", end_token="</Video>",
        add_second_msg=True,
        random_shuffle=True,
    ):
        super().__init__(
            ann_file, transform,
            system=system, role=role,
            start_token=start_token, end_token=end_token,
            random_shuffle=random_shuffle,
        )
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.add_second_msg = add_second_msg

        logger.info(f"Use {video_reader_type} for data in {ann_file}")
        if add_second_msg:
            logger.info(f"Add second message: The video contains X frames sampled at T seconds.")

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            msg = ""
            clip = None
            if "start" in ann and "end" in ann:
                clip = [ann["start"], ann["end"]]
            video, index, sec = self.load_and_transform_media_data_video(index, ann["image"], return_fps=True, clip=clip)
            if self.add_second_msg:
                # " " should be added in the start and end
                msg = f" The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds. "
            conversation, instruction, question, generate, answer = self.process_qa(ann["qa"], msg)
            question_type = ann["question_type"]
            return video, conversation, instruction, question, index, generate, answer, question_type
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading video {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)