#!/bin/bash

pushd ..
rm -rf ./submission_files.zip
zip -r ./submission_files.zip ./src ./third-party ./epoch-4-step-31725-012914.ckpt ./epoch-4-step-50850-012821.ckpt
popd
