$IMAGE_NAME="kaggle/nfl:0.0.1"
pushd ..
docker run --rm -v "${pwd}:/workspace" `
     -v ~/.ssh:/home/docker_user/.ssh:ro `
     -w /workspace `
     -it `
     --entrypoint /bin/bash `
     -p 8888:8888 `
     -p 6006:6006 `
     --gpus '"device=0"' `
     --shm-size=8gb `
     $IMAGE_NAME
     # --memory=10g `
popd