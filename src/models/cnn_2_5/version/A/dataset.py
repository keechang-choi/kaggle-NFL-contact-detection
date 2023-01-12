import pytorch_lightning as pl
import random
import pandas as pd
import numpy as np
from torch.utils.data import random_split, Dataset, DataLoader, Subset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm
import glob
import subprocess
import torch
import gc
from typing import Tuple, List, Optional
import pickle

from config import CFG


class CNN25Dataset(Dataset):
    def __init__(self, df, data_dir, tmp_data_dir, feature_cols, video2helmets, video2frames, aug, mode='train'):
        self.df = df
        self.data_dir = data_dir
        # kaggle read only dir 문제로, 임시 생성 dir 별도 지정.
        self.tmp_data_dir = tmp_data_dir
        self.frame = df.frame.values
        self.feature = df[feature_cols].fillna(-1).values
        self.players = df[['nfl_player_id_1', 'nfl_player_id_2']].values
        self.game_play = df.game_play.values
        self.aug = aug
        self.mode = mode

        self.video2helmets = video2helmets
        self.video2frames = video2frames

    def __len__(self):
        return len(self.df)

    # @lru_cache(1024)
    # def read_img(self, path):
    #     return cv2.imread(path, 0)

    def __getitem__(self, idx):
        window = 24
        frame = self.frame[idx]

        # TODO: 이 부분 의미 잘 모르겠음. (기존코드)
        if self.mode in ["fit", "validate", "test"]:
            frame = frame + random.randint(-6, 6)

        players = []
        for p in self.players[idx]:
            if p == 'G':
                players.append(p)
            else:
                players.append(int(p))

        imgs = []
        for view in ['Endzone', 'Sideline']:
            video = self.game_play[idx] + f'_{view}.mp4'

            tmp = self.video2helmets[video]
#             tmp = tmp.query('@frame-@window<=frame<=@frame+@window')
            tmp[tmp['frame'].between(frame-window, frame+window)]
            # .sort_values(['nfl_player_id', 'frame'])
            tmp = tmp[tmp.nfl_player_id.isin(players)]
            tmp_frames = tmp.frame.values
            tmp = tmp.groupby('frame')[
                ['left', 'width', 'top', 'height']].mean()
# 0.002s

            bboxes = []
            for f in range(frame-window, frame+window+1, 1):
                if f in tmp_frames:
                    x, w, y, h = tmp.loc[f][['left', 'width', 'top', 'height']]
                    bboxes.append([x, w, y, h])
                else:
                    bboxes.append([np.nan, np.nan, np.nan, np.nan])
            bboxes = pd.DataFrame(bboxes).interpolate(
                limit_direction='both').values
            bboxes = bboxes[::4]

            if bboxes.sum() > 0:
                flag = 1
            else:
                flag = 0
# 0.03s

            for i, f in enumerate(range(frame-window, frame+window+1, 4)):
                img_new = np.zeros((256, 256), dtype=np.float32)

                if flag == 1 and f <= self.video2frames[video]:
                    img = cv2.imread(
                        os.path.join(self.tmp_data_dir, f"frames/{video}_{f:04d}.jpg"), 0)

                    x, w, y, h = bboxes[i]

                    img = img[int(y+h/2)-128:int(y+h/2)+128,
                              int(x+w/2)-128:int(x+w/2)+128].copy()
                    img_new[:img.shape[0], :img.shape[1]] = img

                imgs.append(img_new)
# 0.06s

        feature = np.float32(self.feature[idx])

        img = np.array(imgs).transpose(1, 2, 0)
        img = self.aug(image=img)["image"]
        label = np.float32(self.df.contact.values[idx])

        return img, feature, label


class CNN25DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", tmp_data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.tmp_data_dir = tmp_data_dir

        self.train_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.Normalize(mean=[0.], std=[1.]),
            ToTensorV2()
        ])

        self.valid_aug = A.Compose([
            A.Normalize(mean=[0.], std=[1.]),
            ToTensorV2()
        ])
        self.use_cols = [
            'x_position', 'y_position', 'speed', 'distance',
            'direction', 'orientation', 'acceleration', 'sa'
        ]
        # TODO: 이 부분 hard-coding 된 것 개선.
        self.feature_cols: List[str] = ["distance", "G_flug"]
        for col in self.use_cols:
            self.feature_cols.append(col + "_1")
            self.feature_cols.append(col + "_2")

        self.dataset_test = None
        self.dataset_train = None
        self.dataset_valid = None
        self.dataset_pred = None

    def expand_contact_id(self, df):
        """
        Splits out contact_id into seperate columns.
        """
        df["game_play"] = df["contact_id"].str[:12]
        df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
        df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
        df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
        return df

    def create_features(self, df, tr_tracking, merge_col="step", use_cols=["x_position", "y_position"]):
        output_cols = []
        df_combo = (
            df.astype({"nfl_player_id_1": "str"})
            .merge(
                tr_tracking.astype({"nfl_player_id": "str"})[
                    ["game_play", merge_col, "nfl_player_id",] + use_cols
                ],
                left_on=["game_play", merge_col, "nfl_player_id_1"],
                right_on=["game_play", merge_col, "nfl_player_id"],
                how="left",
            )
            .rename(columns={c: c+"_1" for c in use_cols})
            .drop("nfl_player_id", axis=1)
            .merge(
                tr_tracking.astype({"nfl_player_id": "str"})[
                    ["game_play", merge_col, "nfl_player_id"] + use_cols
                ],
                left_on=["game_play", merge_col, "nfl_player_id_2"],
                right_on=["game_play", merge_col, "nfl_player_id"],
                how="left",
            )
            .drop("nfl_player_id", axis=1)
            .rename(columns={c: c+"_2" for c in use_cols})
            .sort_values(["game_play", merge_col, "nfl_player_id_1", "nfl_player_id_2"])
            .reset_index(drop=True)
        )
        output_cols += [c+"_1" for c in use_cols]
        output_cols += [c+"_2" for c in use_cols]

        if ("x_position" in use_cols) & ("y_position" in use_cols):
            index = df_combo['x_position_2'].notnull()

            distance_arr = np.full(len(index), np.nan)
            tmp_distance_arr = np.sqrt(
                np.square(df_combo.loc[index, "x_position_1"] -
                          df_combo.loc[index, "x_position_2"])
                + np.square(df_combo.loc[index, "y_position_1"] -
                            df_combo.loc[index, "y_position_2"])
            )

            distance_arr[index] = tmp_distance_arr
            df_combo['distance'] = distance_arr
            output_cols += ["distance"]

        df_combo['G_flug'] = (df_combo['nfl_player_id_2'] == "G")
        output_cols += ["G_flug"]
        return df_combo, output_cols

    def prepare_data(self):
        # 다운로드 및 전처리
        # NOTE: single core로 실행됨.
        # https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html
        frame_dir = os.path.join(self.tmp_data_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        # NOTE: state 저장하지 말고 disk에 저장해야함.
        self.preprocess_dataset()

    def preprocess_dataset(self):
        # 데이터 전처리 후, 파일로 저장해놓는다.
        # fit,validate,test 공통으로 한번, predict 한번씩만 실행되면 된다.
        is_prediction = CFG["is_prediction"]
        print(f"Preprocess ffmpeg. is_prediction: {is_prediction}")

        if is_prediction:
            file_name = "test"
        else:
            file_name = "train"

        # ffmpeg 데이터 전처리
        frame_dir = os.path.join(self.tmp_data_dir, "frames")
        df_helmets = pd.read_csv(os.path.join(
            self.data_dir, f"{file_name}_baseline_helmets.csv"))
        print(f"-- ffmpeg frames {file_name}")
        for video in tqdm(df_helmets.video.unique()):
            if os.path.isfile(os.path.join(frame_dir, video+"_0001.jpg")):
                continue
            if "Endzone2" not in video:
                subprocess.call(["ffmpeg", "-i", os.path.join(self.data_dir, f"{file_name}/{video}"), "-q:v", "2", "-f", "image2", os.path.join(
                    frame_dir, f"{video}_%04d.jpg"), "-hide_banner", "-loglevel", "error"])

        df_video_metadata = pd.read_csv(os.path.join(
            self.data_dir, f"{file_name}_video_metadata.csv"))

        print(
            f"-- video mapping {file_name}: {len(df_helmets.video.unique())}")
        video2helmets = {}
        df_helmets_new = df_helmets.set_index('video')
        for video in tqdm(df_helmets.video.unique()):
            video2helmets[video] = df_helmets_new.loc[video].reset_index(
                drop=True)

        video2frames = {}
        for game_play in tqdm(df_video_metadata.game_play.unique()):
            for view in ['Endzone', 'Sideline']:
                video = game_play + f'_{view}.mp4'
                video2frames[video] = max(list(map(
                    lambda x: int(x.split('_')[-1].split('.')[0]),
                    glob.glob(os.path.join(self.tmp_data_dir, f'frames/{video}*')
                              ))))
        # 메모리 이슈
        del df_helmets, df_helmets_new
        gc.collect()

        print(f"-- Generating dataframe from file : {file_name}")
        df_tracking = pd.read_csv(os.path.join(
            self.data_dir, f"{file_name}_player_tracking.csv"))

        if file_name == "test":
            label_file_name = "sample_submission.csv"
        else:
            label_file_name = "train_labels.csv"

        labels = self.expand_contact_id(pd.read_csv(
            os.path.join(self.data_dir, label_file_name)))

        df_with_feature, _ = self.create_features(
            labels, df_tracking, use_cols=self.use_cols)

        df_filtered = df_with_feature.query(
            'not distance>2').reset_index(drop=True)
        df_filtered['frame'] = (
            df_filtered['step']/10*59.94+5*59.94).astype('int')+1

        # 메모리 이슈
        del df_with_feature, labels, df_tracking
        gc.collect()

        # save preprocessed files to writable dir.
        df_filtered.to_csv(os.path.join(self.tmp_data_dir, "df_filtered.csv"))
        with open(os.path.join(self.tmp_data_dir, "video2helmets.pickle"), "wb") as f:
            pickle.dump(video2helmets, f)
        with open(os.path.join(self.tmp_data_dir, "video2frames.pickle"), "wb") as f:
            pickle.dump(video2frames, f)

    def generate_dataset(self, stage: str) -> CNN25Dataset:
        # 학습 데이터 split을 수행한다.

        print(f"Generating dataset: {stage}")

        df_filtered = pd.read_csv(os.path.join(
            self.tmp_data_dir, "df_filtered.csv"))
        feature_cols = self.feature_cols
        with open(os.path.join(self.tmp_data_dir, "video2helmets.pickle"), "rb") as f:
            video2helmets = pickle.load(f)
        with open(os.path.join(self.tmp_data_dir, "video2frames.pickle"), "rb") as f:
            video2frames = pickle.load(f)

        if stage in ["fit", "validate", "test"]:
            # TODO: config 바로 이용하는게낫나, args로 넘기는게 낫나
            game_play_arr: np.ndarray
            game_play_arr = df_filtered.groupby(
                "game_play")["game_play"].first().to_numpy()
            # 사용할 game_play 수를 줄여준다.
            game_play_arr = game_play_arr[:CFG["num_train_video"]]

            train_video_size = int(len(game_play_arr) * 0.8)
            valid_video_size = int(len(game_play_arr) * 0.1)
            test_video_size = len(game_play_arr) - \
                (train_video_size + valid_video_size)
            # TODO: seed 고정을 함수 밖으로 옮기는게 나을지
            seed = torch.Generator().manual_seed(CFG["seed"])
            train_split, valid_split, test_split = random_split(
                game_play_arr, [train_video_size, valid_video_size, test_video_size], generator=seed)

            if stage == "fit":
                game_play_filter_arr = game_play_arr[train_split.indices]
            elif stage == "validate":
                game_play_filter_arr = game_play_arr[valid_split.indices]
            else:
                # test
                game_play_filter_arr = game_play_arr[test_split.indices]
            print(
                f"-- num_videos: {len(game_play_filter_arr)}/{len(game_play_arr)}")
            df_filtered_dataset = df_filtered[df_filtered["game_play"].isin(
                game_play_filter_arr)]
        else:
            df_filtered_dataset = df_filtered

        # NOTE: predict에서는 split 및 데이터 줄이는 과정이 필요없다.

        print(f"-- Label count {stage}: ")
        print(df_filtered_dataset.groupby("contact")["contact"].count())

        dataset = CNN25Dataset(
            df=df_filtered_dataset,
            data_dir=self.data_dir,
            tmp_data_dir=self.tmp_data_dir,
            feature_cols=feature_cols,
            video2helmets=video2helmets,
            video2frames=video2frames,
            aug=self.valid_aug,
            mode=stage)

        return dataset

    def setup(self, stage: str):
        # dataset 생성
        # stage 는 fit/validate/test/predict 중 하나임.
        # train_ 데이터를 다시 train/validation/test로 나누고,
        # test_ 데이터는 predict에 사용함.

        # https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html
        # 데이터셋 생성

        if stage == "fit":
            self.dataset_train = self.generate_dataset(stage)
            print(f"{stage}_dataset_size: {len(self.dataset_train)}")
        elif stage == "validate":
            self.dataset_valid = self.generate_dataset(stage)
            print(f"{stage}_dataset_size: {len(self.dataset_valid)}")
        elif stage == "test":
            self.dataset_test = self.generate_dataset(stage)
            print(f"{stage}_dataset_size: {len(self.dataset_test)}")
        elif stage == "predict":
            self.dataset_pred = self.generate_dataset(stage)
            print(f"{stage}_dataset_size: {len(self.dataset_pred)}")
        else:
            raise TypeError("stage error.")

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"])

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"])

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"])

    def predict_dataloader(self):
        return DataLoader(self.dataset_pred, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"])
