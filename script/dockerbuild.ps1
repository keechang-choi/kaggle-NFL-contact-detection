$IMAGE_NAME="kaggle/nfl:0.0.1"
pushd ..
docker build --force-rm -t $IMAGE_NAME . -f Dockerfile
popd
