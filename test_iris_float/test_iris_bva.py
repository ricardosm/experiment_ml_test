import json
import pytest
from hypothesis import given, strategies as st, settings, Phase
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from joblib import load


@pytest.fixture(autouse=True, scope='class')
def _setup(request):
    request.cls.model = load(
        'C:/Users/Ricardo/Downloads/00_mestrado/dissertation_project/examples/iris/test_iris_float/decision_tree_model.joblib')

    request.cls.data = dict()
    request.cls.data['n_test'] = 8
    request.cls.data['n_samples_per_test'] = 100
    request.cls.data['tests'] = dict()

    for i in range(request.cls.data['n_test']):
        teste_id = 'test_' + str(i + 1)
        request.cls.data['tests'][teste_id] = {'n_samples': 0, 'samples': [], 'y_expected': [], 'y_predicted': []}

    path = 'C:/Users/Ricardo/Downloads/00_mestrado/dissertation_project/examples/iris/test_iris_float/test_iris_bva_experiment_data.json'
    yield path
    with open(path, mode='w') as json_file:
        json.dump(request.cls.data, json_file)


class TestIrisProperty:

    @given(st.sampled_from([4.9, 5.6, 5.7, 5.8, 6.0, 6.1, 6.3, 6.5, 6.9, 7.7]),
           st.sampled_from([2.2, 2.6, 2.7, 2.8, 2.9, 3.1, 3.3, 3.4, 3.6, 3.8]),
           st.floats(min_value=4.850001, max_value=5.26, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.650001, max_value=1.82, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_1(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_1']['n_samples'] += 1
        self.data['tests']['test_1']['samples'].append(x_test)
        self.data['tests']['test_1']['y_expected'].append(y_expected[0])
        self.data['tests']['test_1']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([4.9, 5.2, 5.4, 5.9, 6.1, 6.3, 6.4, 6.5, 6.7, 6.8]),
           st.floats(min_value=3.100001, max_value=3.36, exclude_min=True, allow_nan=False),
           st.floats(min_value=2.45, max_value=2.93, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.650001, max_value=1.82, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_2(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_2']['n_samples'] += 1
        self.data['tests']['test_2']['samples'].append(x_test)
        self.data['tests']['test_2']['y_expected'].append(y_expected[0])
        self.data['tests']['test_2']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([5.6, 5.9, 6.0, 6.1, 6.2, 6.3, 6.7, 6.9, 7.2, 7.9]),
           st.floats(min_value=2.88, max_value=3.1, allow_nan=False),
           st.floats(min_value=2.45, max_value=2.93, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.650001, max_value=1.82, exclude_min=True, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_3(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_3']['n_samples'] += 1
        self.data['tests']['test_3']['samples'].append(x_test)
        self.data['tests']['test_3']['y_expected'].append(y_expected[0])
        self.data['tests']['test_3']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=6.150001, max_value=6.5, exclude_min=True, allow_nan=False),
           st.sampled_from([2.2, 2.5, 2.6, 2.8, 2.9, 3.1, 3.2, 3.3, 3.4, 3.8]),
           st.floats(min_value=4.950001, max_value=5.34, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.34, max_value=1.65, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_4(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_4']['n_samples'] += 1
        self.data['tests']['test_4']['samples'].append(x_test)
        self.data['tests']['test_4']['y_expected'].append(y_expected[0])
        self.data['tests']['test_4']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.78, max_value=6.15, allow_nan=False),
           st.floats(min_value=2.450001, max_value=2.84, exclude_min=True, allow_nan=False),
           st.floats(min_value=4.950001, max_value=5.34, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.34, max_value=1.65, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_5(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_5']['n_samples'] += 1
        self.data['tests']['test_5']['samples'].append(x_test)
        self.data['tests']['test_5']['y_expected'].append(y_expected[0])
        self.data['tests']['test_5']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.floats(min_value=5.78, max_value=6.15, allow_nan=False),
           st.floats(min_value=2.360001, max_value=2.45, allow_nan=False),
           st.floats(min_value=4.950001, max_value=5.34, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.34, max_value=1.65, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_6(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [2]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_6']['n_samples'] += 1
        self.data['tests']['test_6']['samples'].append(x_test)
        self.data['tests']['test_6']['y_expected'].append(y_expected[0])
        self.data['tests']['test_6']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([4.9, 5.0, 5.2, 5.4, 5.6, 6.2, 6.4, 6.8, 6.9, 7.0]),
           st.sampled_from([2.3, 2.5, 2.6, 2.7, 2.8, 3.0, 3.1, 3.2, 3.3, 3.4]),
           st.floats(min_value=2.45, max_value=2.95, exclude_min=True, allow_nan=False),
           st.floats(min_value=1.34, max_value=1.65, allow_nan=False))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_7(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [1]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_7']['n_samples'] += 1
        self.data['tests']['test_7']['samples'].append(x_test)
        self.data['tests']['test_7']['y_expected'].append(y_expected[0])
        self.data['tests']['test_7']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted

    @given(st.sampled_from([4.3, 4.4, 4.5, 4.6, 4.7, 4.9, 5.0, 5.3, 5.5, 5.7]),
           st.sampled_from([2.3, 2.9, 3.1, 3.2, 3.3, 3.4, 3.5, 3.7, 3.8, 4.4]),
           st.floats(min_value=2.16, max_value=2.449999, allow_nan=False),
           st.sampled_from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    @settings(phases=[Phase.generate], max_examples=100)
    def test_8(self, feature_1, feature_2, feature_3, feature_4):
        x_test = [feature_1, feature_2, feature_3, feature_4]
        y_expected = [0]
        y_predicted = self.model.predict([x_test]).tolist()

        self.data['tests']['test_8']['n_samples'] += 1
        self.data['tests']['test_8']['samples'].append(x_test)
        self.data['tests']['test_8']['y_expected'].append(y_expected[0])
        self.data['tests']['test_8']['y_predicted'].append(y_predicted[0])

        assert y_expected == y_predicted
