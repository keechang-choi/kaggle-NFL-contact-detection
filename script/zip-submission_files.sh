#!/bin/bash

pushd ..
rm -rf ./submission_files.zip
zip -r ./submission_files.zip ./src ./third-party ./epoch-9-model-C-players-020221.ckpt ./epoch-8-model-C-ground-only-020220.ckpt
popd
