#!/bin/bash

pushd ..
rm -rf ./submission_files.zip
zip -r ./submission_files.zip ./src ./third-party ./epoch-9-step-220200.ckpt
popd
