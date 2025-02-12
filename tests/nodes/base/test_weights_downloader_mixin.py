# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import logging
import tempfile
from pathlib import Path
from unittest import TestCase, mock

import pytest
import requests
import yaml

from peekingduck.nodes.base import PEEKINGDUCK_WEIGHTS_SUBDIR, WeightsDownloaderMixin
from tests.conftest import PKD_DIR, assert_msg_in_logs, do_nothing

# cannot change weights directory for these models, should be skipped for all tests
SKIPPED_MODELS = ["mediapipe_hub"]
# these weights are not on GCS, should be skipped for integrity tests
THIRD_PARTY_MODELS = SKIPPED_MODELS + ["huggingface_hub"]


@pytest.fixture(
    name="weights_model",
    params=[
        path
        for path in (PKD_DIR / "configs" / "model").glob("*.yml")
        if path.stem not in SKIPPED_MODELS
    ],
)
def fixture_weights_model(request):
    return WeightsModel(request.param)


@pytest.fixture(
    name="shipped_weights_model",
    params=[
        path
        for path in (PKD_DIR / "configs" / "model").glob("*.yml")
        if path.stem not in THIRD_PARTY_MODELS
    ],
)
def fixture_shipped_weights_model(request):
    """Includes only models whose weights files are stored on GCS. Excludes
    model hub models.
    """
    return WeightsModel(request.param)


@pytest.fixture(name="weights_type_model", params=["csrnet", "yolox"])
def fixture_weights_type_model(request):
    """Selects 2 models with different weights structure:
    - csrnet: SavedModel format (a folder of files)
    - yolox: PyTorch (single weights file)
    """
    return WeightsModel(PKD_DIR / "configs" / "model" / f"{request.param}.yml")


class WeightsModel(WeightsDownloaderMixin):
    def __init__(self, config_file):
        with open(config_file) as infile:
            node_config = yaml.safe_load(infile)
            node_config["root"] = PKD_DIR
        self.config = node_config
        self.logger = logging.getLogger("test_weights_downloader_mixin.WeightsModel")


class TestWeightsDownloaderMixin:
    def test_parent_dir_not_exist(self, weights_model):
        invalid_dir = "invalid_dir"
        weights_model.config["weights_parent_dir"] = invalid_dir

        with pytest.raises(FileNotFoundError) as excinfo:
            weights_model.download_weights()
        assert f"weights_parent_dir does not exist: {invalid_dir}" == str(excinfo.value)

    def test_parent_dir_not_absolute(self, weights_model):
        relative_dir = PKD_DIR.relative_to(PKD_DIR.parent)
        weights_model.config["weights_parent_dir"] = relative_dir

        with pytest.raises(ValueError) as excinfo:
            weights_model.download_weights()
        assert f"weights_parent_dir must be an absolute path: {relative_dir}" == str(
            excinfo.value
        )

    def test_default_parent_dir(self, weights_model):
        """Checks that _find_paths() gives the correct path when
        `weights_parents_dir=None`.
        """
        parent_dir = Path.cwd().resolve().parent
        weights_model.config["weights_parent_dir"] = parent_dir
        actual = weights_model._find_paths()
        assert (
            actual
            == parent_dir
            / PEEKINGDUCK_WEIGHTS_SUBDIR
            / weights_model.config["weights"][weights_model.config["model_format"]][
                "model_subdir"
            ]
            / weights_model.config["model_format"]
        )

    def test_custom_parent_dir(self, weights_model):
        """Checks that _find_paths() gives the correct path when
        `weights_parents_dir` is a valid custom path.
        """
        actual = weights_model._find_paths()
        assert (
            actual
            == weights_model.config["root"].parent
            / PEEKINGDUCK_WEIGHTS_SUBDIR
            / weights_model.config["weights"][weights_model.config["model_format"]][
                "model_subdir"
            ]
            / weights_model.config["model_format"]
        )

    def test_weights_not_found(self, shipped_weights_model):
        """Checks that the proper logging message is shown then weights are not
        found.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, TestCase.assertLogs(
            "test_weights_downloader_mixin.WeightsModel"
        ) as captured:
            shipped_weights_model.config["weights_parent_dir"] = tmp_dir
            model_dir = shipped_weights_model._find_paths()
            assert not shipped_weights_model._has_weights(model_dir)
            assert_msg_in_logs("No weights detected.", captured.records)

    def test_corrupted_weights(self, shipped_weights_model):
        """Checks that the proper logging message is shown then weights are not
        found.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, TestCase.assertLogs(
            "test_weights_downloader_mixin.WeightsModel"
        ) as captured:
            shipped_weights_model.config["weights_parent_dir"] = tmp_dir
            model_dir = shipped_weights_model._find_paths()
            # Create a temp weights file which doesn't match the checksum
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / shipped_weights_model.model_filename).touch()

            assert not shipped_weights_model._has_weights(model_dir)
            assert_msg_in_logs(
                "Weights file is corrupted/out-of-date.", captured.records
            )

    def test_skip_integrity_check(self, shipped_weights_model):
        """Checks that the proper logging message is shown then weights are not
        found.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, TestCase.assertLogs(
            "test_weights_downloader_mixin.WeightsModel"
        ) as captured, mock.patch.object(
            WeightsDownloaderMixin, "_get_weights_checksum"
        ) as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError()
            shipped_weights_model.config["weights_parent_dir"] = tmp_dir
            model_dir = shipped_weights_model._find_paths()
            # Create a temp weights file which doesn't match the checksum
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / shipped_weights_model.model_filename).touch()

            assert shipped_weights_model._has_weights(model_dir)
            assert_msg_in_logs(
                "Skipped weights file integrity check due to network connectivity error.",
                captured.records,
            )

    def test_valid_weights(self, weights_type_model):
        """Checks that verifying weights checksum works for both single weights
        file and a weights directory (SavedModel format).

        Currently, `weights_type_model` only hold 2 models to reduce bandwidth
        usage.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_type_model.config["weights_parent_dir"] = tmp_dir
            model_dir = weights_type_model._find_paths()
            model_dir.mkdir(parents=True, exist_ok=True)

            weights_type_model._download_to(weights_type_model.blob_filename, model_dir)
            weights_type_model._extract_file(model_dir)

            assert weights_type_model._has_weights(model_dir)

    def test_sha256sum_ignores_macos_files(self):
        """Checks that extra files created on Mac OS is ignored by the
        sha256sum() method.
        """
        all_files = [".DS_Store", "__MACOSX", "file1", "file2"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_directory = Path(tmp_dir)
            for i, file in enumerate(sorted(all_files)):
                (tmp_directory / file).write_text(str(i))

            expected = hashlib.sha256()
            for file in sorted(all_files[2:]):
                expected = WeightsDownloaderMixin.sha256sum(
                    tmp_directory / file, expected
                )
            assert (
                WeightsDownloaderMixin.sha256sum(tmp_directory).hexdigest()
                == expected.hexdigest()
            )

    @pytest.mark.usefixtures("tmp_dir")
    @mock.patch.object(WeightsDownloaderMixin, "_download_to", wraps=do_nothing)
    @mock.patch.object(WeightsDownloaderMixin, "_extract_file", wraps=do_nothing)
    def test_create_weights_dir(
        self, mock_download_to, mock_extract_file, shipped_weights_model
    ):
        with tempfile.TemporaryDirectory() as tmp_dir, TestCase.assertLogs(
            "test_weights_downloader_mixin.WeightsModel"
        ) as captured:
            weights_parent_dir = Path(tmp_dir)
            model_dir = (
                weights_parent_dir
                / PEEKINGDUCK_WEIGHTS_SUBDIR
                / shipped_weights_model.model_subdir
                / shipped_weights_model.config["model_format"]
            )
            shipped_weights_model.config["weights_parent_dir"] = weights_parent_dir

            assert not (weights_parent_dir / PEEKINGDUCK_WEIGHTS_SUBDIR).exists()

            shipped_weights_model.download_weights()

            assert mock_download_to.called
            assert mock_extract_file.called
            assert (weights_parent_dir / PEEKINGDUCK_WEIGHTS_SUBDIR).exists()

            assert_msg_in_logs("No weights detected.", captured.records)
            assert_msg_in_logs("Proceeding to download...", captured.records)
            assert_msg_in_logs(f"Weights downloaded to {model_dir}.", captured.records)
