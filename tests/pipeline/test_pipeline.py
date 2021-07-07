"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import pytest
from peekingduck.pipeline.pipeline import Pipeline
from peekingduck.pipeline.nodes.node import AbstractNode


class MockedNode(AbstractNode):
    def __init__(self, config):
        super().__init__(config, node_path=__name__)

    def run(self, inputs):
        output = {}
        for idx in range(len(self._outputs)):
            output[self._outputs[idx]] = "test_output_" + str(idx)

        print(output)
        return output


@pytest.fixture
def config_node_input():
    return {'input': ["none"],
            'output': ["test_output_1"]}


@pytest.fixture
def config_node_end():
    return {'input': ["test_output_1"],
            'output': ["test_output_2", "pipeline_end"]}


@pytest.fixture
def test_input_node(config_node_input):
    return MockedNode(config_node_input)


@pytest.fixture
def test_node_end(config_node_end):
    return MockedNode(config_node_end)


@pytest.fixture
def pipeline_correct(test_input_node, test_node_end):
    return Pipeline([test_input_node, test_node_end])


class TestPipeline:
    def test_execute(self, pipeline_correct):
        correct_data = {'test_output_1': 'test_output_0', 
                        'test_output_2': 'test_output_0', 
                        'pipeline_end': 'test_output_1'}
        pipeline_correct.execute()

        assert pipeline_correct._data == correct_data
        assert pipeline_correct.get_pipeline_results() == correct_data

    def test_pipeline_wrong_order(self, test_input_node, test_node_end):
        with pytest.raises(ValueError):
            Pipeline([test_node_end, test_input_node])

    def test_empty_pipeline_results(self, pipeline_correct):
        assert not pipeline_correct.get_pipeline_results()
