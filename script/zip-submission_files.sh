#!/bin/bash

pushd ..
rm -rf ./submission_files.zip
zip -r ./submission_files.zip ./src ./third-party ./epoch-8-step-35289-011716.ckpt
popd
