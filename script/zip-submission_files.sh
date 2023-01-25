#!/bin/bash

pushd ..
rm -rf ./submission_files.zip
zip -r ./submission_files.zip ./src ./third-party ./epoch-8-step-198180-012222.ckpt
popd
