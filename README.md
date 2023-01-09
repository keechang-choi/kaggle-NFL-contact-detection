# kaggle-NFL-contact-detection

## docker
- docker image
  - ./dockerbuild.sh (or DOCKER_BUILDKIT=0 ./dockerbuild.sh)
  - ./dockerrun.sh
- TODO: devcontainer
- TODO: jupyter notebook
  - 집에 데탑 port 설정 안해놔서 못하는 중
## data
  - pre-process
    - ffmpeg로 영상 frame 별로 나눠서 jpg로 저장
    - ``` python preprocess-ffmpeg.py ```
