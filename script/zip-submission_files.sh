#!/bin/bash

pushd ..
rm -rf ./submission_files.zip
zip -r ./submission_files.zip ./src ./third-party ./epoch-9-step-39210-011916.ckpt
popd
