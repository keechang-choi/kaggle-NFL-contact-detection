import pytorch_lightning as pl
import random
import pandas as pd
import numpy as np
from torch.utils.data import random_split, Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm
import glob
import subprocess
from config import CFG


class MyDataset(Dataset):
    def __init__(self, df, data_dir, feature_cols, video2helmets, video2frames, aug, mode='train'):
        self.df = df
        self.data_dir = data_dir
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

        if self.mode == 'train':
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
                        os.path.join(self.data_dir, f"frames/{video}_{f:04d}.jpg"), 0)

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


class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir

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
        self.dataset_test = None

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

        test_helmets = pd.read_csv(os.path.join(
            self.data_dir, "test_baseline_helmets.csv"))
        frame_dir = os.path.join(self.data_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        print(f"test videos: {len(test_helmets)}")
        for video in tqdm(test_helmets.video.unique()):
            if os.path.isfile(os.path.join(frame_dir, video+"_0001.jpg")):
                continue
            if "Endzone2" not in video:
                subprocess.call(["ffmpeg", "-i", os.path.join(self.data_dir, f"test/{video}"), "-q:v", "2", "-f", "image2", os.path.join(
                    frame_dir, f"{video}_%04d.jpg"), "-hide_banner", "-loglevel", "error"])

        train_helmets = pd.read_csv(os.path.join(
            self.data_dir, "train_baseline_helmets.csv"))
        os.makedirs(frame_dir, exist_ok=True)
        print(f"train videos: {len(train_helmets)}")
        for video in tqdm(train_helmets.video.unique()):
            if os.path.isfile(os.path.join(frame_dir, video+"_0001.jpg")):
                continue
            # Endzone2 가 view 중에 있는데 뭔지 모름 파악 필요.
            if "Endzone2" not in video:
                subprocess.call(["ffmpeg", "-i", os.path.join(self.data_dir, f"train/{video}"), "-q:v", "2", "-f", "image2", os.path.join(
                    frame_dir, f"{video}_%04d.jpg"), "-hide_banner", "-loglevel", "error"])

    def setup(self, stage: str):
        # dataset 생성

        test_helmets = pd.read_csv(os.path.join(
            self.data_dir, "test_baseline_helmets.csv"))
        test_tracking = pd.read_csv(os.path.join(
            self.data_dir, "test_player_tracking.csv"))
        test_video_metadata = pd.read_csv(os.path.join(
            self.data_dir, "test_video_metadata.csv"))

        labels = self.expand_contact_id(pd.read_csv(
            os.path.join(self.data_dir, "sample_submission.csv")))

        use_cols = [
            'x_position', 'y_position', 'speed', 'distance',
            'direction', 'orientation', 'acceleration', 'sa'
        ]

        test, feature_cols = self.create_features(
            labels, test_tracking, use_cols=use_cols)

        test_filtered = test.query('not distance>2').reset_index(drop=True)
        test_filtered['frame'] = (
            test_filtered['step']/10*59.94+5*59.94).astype('int')+1

        video2helmets = {}
        test_helmets_new = test_helmets.set_index('video')
        for video in tqdm(test_helmets.video.unique()):
            video2helmets[video] = test_helmets_new.loc[video].reset_index(
                drop=True)

        video2frames = {}
        for game_play in tqdm(test_video_metadata.game_play.unique()):
            for view in ['Endzone', 'Sideline']:
                video = game_play + f'_{view}.mp4'
                video2frames[video] = max(list(map(
                    lambda x: int(x.split('_')[-1].split('.')[0]),
                    glob.glob(os.path.join(self.data_dir, f'frames/{video}*')
                              ))))
        self.dataset_test = MyDataset(
            df=test_filtered,
            data_dir=self.data_dir,
            feature_cols=feature_cols,
            video2helmets=video2helmets,
            video2frames=video2frames,
            aug=self.valid_aug,
            mode='test')

    def train_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"])

    def val_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"])

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"])

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"])
