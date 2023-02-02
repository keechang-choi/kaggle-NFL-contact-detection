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


class CNN25SingleGroundDataset(Dataset):
    def __init__(self, df, data_dir, preprocess_result_dir, feature_cols, video2frames, aug, mode='train'):
        self.df = df
        self.data_dir = data_dir
        # kaggle read only dir 문제로, 임시 생성 dir 별도 지정.
        self.preprocess_result_dir = preprocess_result_dir
        self.frame = df.frame.values
        self.feature = df[feature_cols].fillna(-1).values
        self.players = df[['nfl_player_id_1', 'nfl_player_id_2']].values
        self.game_play = df.game_play.values
        self.aug = aug
        self.mode = mode

        self.video2frames = video2frames
        # resnet 원래 사이즈로 맞춰줌.
        self.image_size = 224
        os.makedirs(self.preprocess_result_dir, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        window = CFG["window"]  # 24
        frame = self.frame[idx]

        # frame 값은 tracking data (10Hz)를 변환해서 얻은 값인데,
        # 이 값에 해당하는 정확한 frame 값이 helmet data에 없을 수 있음
        players = []
        for p in self.players[idx]:
            if p == 'G':
                players.append(p)
            else:
                players.append(str(p))

        imgs = []
        for view in ['Endzone', 'Sideline']:
            video = self.game_play[idx] + f'_{view}.mp4'
            query_game_play = self.game_play[idx]

            # 하나의 데이터셋은 하나의 이미지를 포함하도록 함.
            query_res = self.df.query(f"game_play == '{query_game_play}' and "
                                      f"nfl_player_id_1 == '{players[0]}' and "
                                      f"nfl_player_id_2 == '{players[1]}' and "
                                      f"frame == {frame}"
                                      )
            is_query_valid = False
            xl, wl, yl, hl = [], [], [], []
            # bbox에 각 선수, view에 대한 frame의 window 평균 헬멧정보를 넣는다.
            for i in range(2):
                if len(query_res) == 1:
                    x = query_res.iloc[0][f"{view}_left_{i+1}"]
                    w = query_res.iloc[0][f"{view}_width_{i+1}"]
                    y = query_res.iloc[0][f"{view}_top_{i+1}"]
                    h = query_res.iloc[0][f"{view}_height_{i+1}"]
                    if all(not np.isnan(value) for value in [x, w, y, h]):
                        xl.append(x)
                        wl.append(w)
                        yl.append(y)
                        hl.append(h)
                        is_query_valid = True

            # 선수 1, 2 중 한명이라도 helmet 정보 있으면 데이터로 씀.
            if is_query_valid:
                flag = 1
                x_avg = sum(xl)/len(xl)
                w_avg = sum(wl)/len(wl)
                y_avg = sum(yl)/len(yl)
                h_avg = sum(hl)/len(hl)
                bbox = [x_avg, w_avg, y_avg, h_avg]
            else:
                flag = 0

            img_new = np.zeros(
                (self.image_size, self.image_size, 3), dtype=np.float32)
            # if frame > self.video2frames[video]:
            # # 목적 frame에 video길이 넘는 경우가 있는데 왜그런지 잘 모르겠음.
            #     print(
            #         f"flag: {flag}, frame: {frame}, video frame: {self.video2frames[video]}")
            if flag == 1 and frame <= self.video2frames[video]:
                try:
                    img = cv2.imread(
                        os.path.join(self.preprocess_result_dir, f"frames/{video}_{frame:04d}.jpg"), cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    x, w, y, h = bbox
                    # print(f"img helmet size: {w} x {h}")
                    # 10~60 정도? helmet size
                    # 헬멧 크기의 10배 정도로 자름.
                    # random으로 헬멧크기의 배수 만큼 자른다.
                    if self.mode == "fit":
                        crop_ratio = np.random.uniform(9.0, 11.0, 1)[0]
                    else:
                        crop_ratio = 10
                    crop_size = int((max(w, h)*crop_ratio))
                    # crop_size = 256
                    img_tmp = np.zeros(
                        (crop_size, crop_size, 3), dtype=np.float32)
                    # print(crop_size)
                    crop_half = crop_size // 2
                    crop_start_y = max(int(y+h/2)-crop_half, 0)
                    crop_end_y = min(int(y+h/2)+crop_half, img.shape[0])

                    crop_start_x = max(int(x+w/2)-crop_half, 0)
                    crop_end_x = min(int(x+w/2)+crop_half, img.shape[1])
                    # crop할 크기에 따라 원본 프레임에서 이미지 잘라냄.
                    img = img[crop_start_y:crop_end_y,
                              crop_start_x:crop_end_x,
                              :]
                    # 자른 후 정규화 및 transform을 추가한다.
                    img = self.aug(image=img)["image"]

                    # 이미지 가장자리에서 crop 크기가 안나오는 경우 있음.
                    offset_y = (crop_size - img.shape[0]) // 2
                    offset_x = (crop_size - img.shape[1]) // 2
                    # resize 하기 전에 zero padding 사용해서 중앙으로 옮김.
                    img_tmp[offset_y: offset_y+img.shape[0],
                            offset_x: offset_x+img.shape[1],
                            :] = img

                    # resize를 통해 CNN에 들어갈 사이즈로 맞춰줌.
                    img_tmp = cv2.resize(
                        img_tmp, (self.image_size, self.image_size))
                    img_new[:img_tmp.shape[0],
                            :img_tmp.shape[1], :] = img_tmp
                except Exception as e:
                    print(os.path.join(self.preprocess_result_dir,
                                       f"frames/{video}_{frame:04d}.jpg"))
                    print(os.path.exists(os.path.join(
                        self.preprocess_result_dir, f"frames/{video}_{frame:04d}.jpg")))
                    print(f"box: {(x,y)}, {(w,h)}")
                    print(f"img is None: {img is None}")
                    print(f"img shape: {img.shape}")
                    print(e)
                    raise e

            imgs.append(img_new)

        feature = np.float32(self.feature[idx])
        imgs = np.array(imgs)
        # view channel(rgb) h w 로 변경해준다.
        imgs = imgs.transpose(0, 3, 1, 2)
        label = np.float32(self.df.contact.values[idx])

        return imgs, feature, label


class CNN25SingleGroundDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", preprocess_result_dir: str = "./", data_filter: str = "all"):
        super().__init__()
        self.data_dir = data_dir
        self.preprocess_result_dir = preprocess_result_dir

        self.train_aug = A.Compose([
            # Bright and Contrast가 의미 있는지 모르겠으나, normalize 안해주면 에러발생.
            A.ToFloat(max_value=255),
            A.HorizontalFlip(p=0.5),  # 숫자 뒤집어 짐
            A.ShiftScaleRotate(p=0.5, rotate_limit=10),  # 45도는 너무 큼.
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=1.0),
        ])

        self.valid_aug = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255.0)
        ])
        self.use_cols = [
            'x_position', 'y_position', 'speed', 'distance',
            'direction', 'orientation', 'acceleration', 'sa'
        ]
        # TODO: 이 부분 hard-coding 된 것 개선.
        self.feature_cols: List[str] = ["G_flug"]
        for col in self.use_cols:
            self.feature_cols.append(col + "_1")
            # NOTE: Ground만 사용할 경우 player1로 feature col 만 사용함
            if data_filter != "ground-only":
                self.feature_cols.append(col + "_2")
        if data_filter != "ground-only":
            self.feature_cols.append("distance")
        # NOTE: 모델의 mlp input feature수도 바뀜, 9 vs 18

        self.dataset_test = None
        self.dataset_train = None
        self.dataset_valid = None
        self.dataset_pred = None
        self.data_filter = data_filter

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
        # NOTE: state 저장하지 말고 disk에 저장해야함.
        self.preprocess_dataset()

    def reindex_by_frame(self, df):
        frames = np.arange(
            df.index.min()-CFG["window"], df.index.max()+CFG["window"])
        game_play = df.iloc[0]["game_play"]
        view = df.iloc[0]["view"]
        nfl_player_id = df.iloc[0]["nfl_player_id"]
        df = df.reindex(frames, fill_value=np.nan)
        df["game_play"] = df["game_play"].fillna(game_play)
        df["view"] = df["view"].fillna(view)
        df["nfl_player_id"] = df["nfl_player_id"].fillna(nfl_player_id)
        df[["left", "width", "top", "height"]] = df[["left", "width",
                                                     "top", "height"]].interpolate(limit_direction="both")
        return df

    def preprocess_dataset(self):
        # 데이터 전처리 후, 파일로 저장해놓는다.
        # fit,validate,test 공통으로 한번, predict 한번씩만 실행되면 된다.
        is_prediction = CFG["is_prediction"]
        run_type = "test" if is_prediction else "train"
        frame_dir = os.path.join(self.preprocess_result_dir, "frames")
        processed_meta_dir = os.path.join(self.preprocess_result_dir, run_type)

        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(processed_meta_dir, exist_ok=True)

        print("====== [Preprocess] ======")
        print(f"- is_prediction: {is_prediction}")
        print(f"- run_type: {run_type}")

        print("------ [Loading Metadata] ------")
        df_helmets = pd.read_csv(os.path.join(
            self.data_dir, f"{run_type}_baseline_helmets.csv"))
        df_video_metadata = pd.read_csv(os.path.join(
            self.data_dir, f"{run_type}_video_metadata.csv"))

        print("------ [ffmpeg] ------")
        print(f"ffmpeg frames {run_type}")
        for video in tqdm(df_helmets.video.unique()):
            if os.path.isfile(os.path.join(frame_dir, video+"_0001.jpg")):
                continue
            if "Endzone2" not in video:
                subprocess.call(["ffmpeg", "-i", os.path.join(self.data_dir, f"{run_type}/{video}"), "-q:v", "2", "-f", "image2", os.path.join(
                    frame_dir, f"{video}_%04d.jpg"), "-hide_banner", "-loglevel", "error"])

        if CFG["reproduce_processed_data"] or not os.path.exists(os.path.join(processed_meta_dir, "video2frames.pickle")):
            print(
                f"-- Mapping video2frames: [size: {len(df_video_metadata.game_play.unique())}]")
            video2frames = {}
            for game_play in tqdm(df_video_metadata.game_play.unique()):
                for view in ['Endzone', 'Sideline']:
                    video = game_play + f'_{view}.mp4'
                    video2frames[video] = max(list(map(
                        lambda x: int(x.split('_')[-1].split('.')[0]),
                        glob.glob(os.path.join(frame_dir, f'{video}*')
                                  ))))
                with open(os.path.join(processed_meta_dir, "video2frames.pickle"), "wb") as f:
                    pickle.dump(video2frames, f)
            # 메모리 이슈
            del video2frames
        else:
            print(f"video2frames already exists.. skip")

        gc.collect()

        print(f"------ [Preprocess helmet sensor data] ------")
        # CFG["reproduce_processed_data"]
        if CFG["reproduce_processed_data"] or not os.path.exists(os.path.join(processed_meta_dir, "df_filtered.csv")):
            df_tracking = pd.read_csv(os.path.join(
                self.data_dir, f"{run_type}_player_tracking.csv"))

            if is_prediction:
                label_file_name = "sample_submission.csv"
            else:
                label_file_name = "train_labels.csv"

            print(f"- Expand contact id")
            labels = self.expand_contact_id(pd.read_csv(
                os.path.join(self.data_dir, label_file_name)))
            print(f"- Create features")
            df_with_feature, _ = self.create_features(
                labels, df_tracking, use_cols=self.use_cols)
            df_filtered = df_with_feature.query(
                'not distance>2').reset_index(drop=True)
            df_filtered['frame'] = (
                df_filtered['step']/10*59.94+5*59.94).astype('int')+1

            # 메모리 이슈
            del df_with_feature, labels, df_tracking
            gc.collect()

            print(f"- Rolling helmet data")
            df_helmets["nfl_player_id"] = df_helmets["nfl_player_id"].astype(
                str)
            # helmet 데이터에 비어있는 frame을 nan으로 채운다.
            df_reindexed_helmets = df_helmets[["game_play", "view", "frame", "nfl_player_id", "left", "width", "top", "height"]]\
                .set_index("frame")\
                .groupby(["game_play", "view", "nfl_player_id"], dropna=False)\
                .apply(self.reindex_by_frame)\
                .reset_index(["game_play", "view", "nfl_player_id"], drop=True)

            df_reindexed_helmets = df_reindexed_helmets.reset_index()

            df_reindexed_helmets["nfl_player_id"] = df_reindexed_helmets["nfl_player_id"].astype(
                str)

            # Endzone2 있는거 제거
            df_reindexed_helmets = df_reindexed_helmets[df_reindexed_helmets["view"] != "Endzone2"]

            # Endzone과 Sideline의 각 xwyh 를 분리한다.
            df_reindexed_helmets = df_reindexed_helmets.pivot_table(values=["left", "width", "top", "height"], index=[
                "game_play", "nfl_player_id", "frame"], columns=["view"], aggfunc="first").reset_index()

            # 컬럼 이름을 다시 지정한다.
            new_columns = []
            for c in df_reindexed_helmets.columns.to_flat_index():
                if '' in c:
                    new_columns.append(c[0])
                else:
                    new_columns.append(c[1]+"_"+c[0])
            df_reindexed_helmets.columns = new_columns
            # 각 컬럼이름
            rename_list = ["nfl_player_id", "Endzone_left", "Endzone_width", "Endzone_top",
                           "Endzone_height", "Sideline_left", "Sideline_width", "Sideline_top", "Sideline_height"]

            # 각 컬럼 이름에 "_1"과 "_2"를 추가해서 rename으로 사용한다.
            for i in range(2):
                rename_dict = {k: k+f"_{i+1}" for k in rename_list}
                df_filtered[f"nfl_player_id_{i+1}"] = df_filtered[f"nfl_player_id_{i+1}"].astype(
                    str)
                # player i+1에 대한 Enndzone, Sideline에서의 헬멧정보
                df_filtered = df_filtered.merge(df_reindexed_helmets.rename(columns=rename_dict),
                                                left_on=[
                                                    "game_play", f"nfl_player_id_{i+1}", "frame"],
                                                right_on=[
                                                    "game_play", f"nfl_player_id_{i+1}", "frame"],
                                                how="left"
                                                )

            # save preprocessed files to writable dir.
            df_filtered.to_csv(os.path.join(
                processed_meta_dir, "df_filtered.csv"), index=False)
        else:
            print("df_filtered already exists.. skip")

    def generate_dataset(self, stage: str) -> CNN25SingleGroundDataset:
        # 학습 데이터 split을 수행한다.

        print(f"====== Generating dataset  ======")
        print(f"- stage: {stage}")

        if stage == "predict":
            run_type = "test"
        else:
            run_type = "train"

        processed_meta_dir = os.path.join(self.preprocess_result_dir, run_type)

        print(f"------ [Load metadata] ------")
        # NOTE: player id type string
        df_filtered = pd.read_csv(os.path.join(
            processed_meta_dir, f"df_filtered.csv"),
            dtype={"nfl_player_id_1": str, "nfl_player_id_2": str})

        with open(os.path.join(processed_meta_dir, "video2frames.pickle"), "rb") as f:
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

        if stage in ["fit"]:
            aug = self.train_aug
        else:
            aug = self.valid_aug

        if self.data_filter == "ground-only":
            # Ground와의 충돌에 관한 데이터만 사용한다.
            df_filtered_dataset = df_filtered_dataset.query(
                'G_flug == True').reset_index(drop=True)
        elif self.data_filter == "players":
            # 선수들 간의 충돌에 관한 데이터만 사용한다.
            df_filtered_dataset = df_filtered_dataset.query(
                'G_flug == False').reset_index(drop=True)

        dataset = CNN25SingleGroundDataset(
            df=df_filtered_dataset,
            data_dir=self.data_dir,
            preprocess_result_dir=self.preprocess_result_dir,
            feature_cols=self.feature_cols,
            video2frames=video2frames,
            aug=aug,
            mode=stage)

        return dataset

    def setup(self, stage: str):
        # dataset 생성
        # stage 는 fit/validate/test/predict 중 하나임.
        # train_ 데이터를 다시 train/validation/test로 나누고,
        # test_ 데이터는 predict에 사용함.

        # https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html
        # 데이터셋 생성
        print("------ [Setup dataset] ------")
        if stage == "fit":
            self.dataset_train = self.generate_dataset("fit")
            print(f"- fit_dataset_size: {len(self.dataset_train)}")
            self.dataset_valid = self.generate_dataset("validate")
            print(f"- validate_dataset_size: {len(self.dataset_valid)}")
        elif stage == "validate":
            self.dataset_valid = self.generate_dataset("validate")
            print(f"- {stage}_dataset_size: {len(self.dataset_valid)}")
        elif stage == "test":
            self.dataset_test = self.generate_dataset("test")
            print(f"- {stage}_dataset_size: {len(self.dataset_test)}")
        elif stage == "predict":
            self.dataset_pred = self.generate_dataset("predict")
            print(f"- {stage}_dataset_size: {len(self.dataset_pred)}")
        else:
            raise TypeError("stage error.")

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"], shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"])

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"])

    def predict_dataloader(self):
        return DataLoader(self.dataset_pred, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"])
