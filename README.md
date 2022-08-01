# Testing ML

Learn how to create reliable ML systems by testing code, data and models.

<div align="left">
    <a target="_blank" href="https://newsletter.madewithml.com"><img src="https://img.shields.io/badge/Subscribe-30K-brightgreen"></a>&nbsp;
    <a target="_blank" href="https://github.com/GokuMohandas/Made-With-ML"><img src="https://img.shields.io/github/stars/GokuMohandas/Made-With-ML.svg?style=social&label=Star"></a>&nbsp;
    <a target="_blank" href="https://www.linkedin.com/in/goku"><img src="https://img.shields.io/badge/style--5eba00.svg?label=LinkedIn&logo=linkedin&style=social"></a>&nbsp;
    <a target="_blank" href="https://twitter.com/GokuMohandas"><img src="https://img.shields.io/twitter/follow/GokuMohandas.svg?label=Follow&style=social"></a>
    <br>
</div>

<br>

👉 &nbsp;This repository is an isolated version of the much more comprehensive [testing lesson](https://madewithml.com/courses/mlops/testing/), which is a part of our free [mlops course](https://madewithml.com/). Use this repository to learn how to test ML systems but be sure to explore the [mlops-course](https://github.com/GokuMohandas/mlops-course) repository to learn how to tie testing workflows with all the other parts of ML system development.

- [Data](#data)
    - [Rows and columns](#rows-and-columns)
    - [Individual values](#individual-values)
    - [Aggregate values](#aggregate-values)
    - [Production](#production)
- [Models](#models)
    - [Training](#training)
    - [Behavioral](#behavioral)
    - [Adversarial](#adversarial)
    - [Inference](#inference)

📓 &nbsp;Run all the code without any setup via our [interactive notebook](https://colab.research.google.com/github/GokuMohandas/mlops-course/blob/main/notebooks/testing.ipynb).

## 🔢&nbsp; Data

Tools such as [pytest](https://madewithml.com/courses/mlops/testing/#pytest) allow us to test the functions that interact with our data but not the validity of the data itself. We're going to use the [great expectations](https://github.com/great-expectations/great_expectations) library to test what our data is expected to look like.

```bash
pip install great-expectations==0.15.15
```

```python
import great_expectations as ge
import json
import pandas as pd
from urllib.request import urlopen
```

```python
# Load projects
url = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.json"
projects = json.loads(urlopen(url).read())
df = ge.dataset.PandasDataset(projects)
print (f"{len(df)} projects")
df.head(5)
```

<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>created_on</th>
      <th>title</th>
      <th>description</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>2020-02-20 06:43:18</td>
      <td>Comparison between YOLO and RCNN on real world...</td>
      <td>Bringing theory to experiment is cool. We can ...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>2020-02-20 06:47:21</td>
      <td>Show, Infer &amp; Tell: Contextual Inference for C...</td>
      <td>The beauty of the work lies in the way it arch...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>2020-02-24 16:24:45</td>
      <td>Awesome Graph Classification</td>
      <td>A collection of important graph embedding, cla...</td>
      <td>graph-learning</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2020-02-28 23:55:26</td>
      <td>Awesome Monte Carlo Tree Search</td>
      <td>A curated list of Monte Carlo tree search papers...</td>
      <td>reinforcement-learning</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>2020-03-03 13:54:31</td>
      <td>Diffusion to Vector</td>
      <td>Reference implementation of Diffusion2Vec (Com...</td>
      <td>graph-learning</td>
    </tr>
  </tbody>
</table>
</div></div>


### Rows and columns

The most basic expectation is validating the presence of samples (rows) and features (columns). These can help identify inconsistencies between upstream backend database schema changes, upstream UI form changes, etc.

- presence of specific features

```python
df.expect_table_columns_to_match_ordered_list(
    column_list=["id", "created_on", "title", "description", "tag"]
)
```

- unique combinations of features (detect data leaks!)

```python
df.expect_compound_columns_to_be_unique(column_list=["title", "description"])
```

- row count (exact or range) of samples

### Individual values

We can also have expectations about the individual values of specific features.

- missing values

```python
df.expect_column_values_to_not_be_null(column="tag")
```

- type adherence (ex. text features are of type `str`)

```python
df.expect_column_values_to_be_of_type(column="title", type_="str")
```

- values must be unique or from a predefined set

```python
df.expect_column_values_to_be_unique(column="id")
```

- list (categorical) / range (continuous) of allowed values
- feature value relationships with other feature values (ex. column 1 values must always be greater than column 2)

### Aggregate values

We can also set expectations about all the values of specific features.

- value statistics (mean, std, median, max, min, sum, etc.)
- distribution shift by comparing current values to previous values (useful for detecting drift)

### Production

The advantage of using a library such as great expectations, as opposed to isolated assert statements is that we can:

- quickly apply tests using the SDK (and create custom ones) as opposed to writing tests from scratch (error prone).
- automatically create testing checkpoints to execute as our dataset grows

```python
# Run all tests on our DataFrame at once
expectation_suite = df.get_expectation_suite(discard_failed_expectations=False)
df.validate(expectation_suite=expectation_suite, only_return_failures=True)
```

This becomes especially crucial as data tests are moved away from individual repositories and applied in [DataOps workflows](https://madewithml.com/courses/mlops/orchestration/#dataops) for many downstream consumers to reliable use.

<img width="700" src="https://madewithml.com/static/images/mlops/testing/production.png" alt="ETL pipelines in production">

## 🤖&nbsp; Models

Once we've tested our data, we can use it for downstream applications such as training machine learning models. It's important that we also test these model artifacts to ensure reliable behavior in our application.

### Training

Unlike traditional software, ML models can run to completion without throwing any exceptions / errors but can produce incorrect systems. We want to catch errors quickly to save on time and compute.

- Check shapes and values of model output
```python
assert model(inputs).shape == torch.Size([len(inputs), num_classes])
```
- Check for decreasing loss after one batch of training
```python
assert epoch_loss < prev_epoch_loss
```
- Overfit on a batch
```python
accuracy = train(model, inputs=batches[0])
assert accuracy == pytest.approx(0.95, abs=0.05) # 0.95 ± 0.05
```
- Train to completion (tests early stopping, saving, etc.)
```python
train(model)
assert learning_rate >= min_learning_rate
assert artifacts
```
- On different devices
```python
assert train(model, device=torch.device("cpu"))
assert train(model, device=torch.device("cuda"))
```

### Behavioral

Behavioral testing is the process of testing input data and expected outputs while treating the model as a black box (model agnostic evaluation). A landmark paper on this topic is [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118) which breaks down behavioral testing into three types of tests:

- `invariance`: Changes should not affect outputs.
```python
# INVariance via verb injection (changes should not affect outputs)
tokens = ["revolutionized", "disrupted"]
texts = [f"Transformers applied to NLP have {token} the ML field." for token in tokens]
predict.predict(texts=texts, artifacts=artifacts)
```
<pre class="output">
['natural-language-processing', 'natural-language-processing']
</pre>
- `directional`: Change should affect outputs.
```python
# DIRectional expectations (changes with known outputs)
tokens = ["text classification", "image classification"]
texts = [f"ML applied to {token}." for token in tokens]
predict.predict(texts=texts, artifacts=artifacts)
```
<pre class="output">
['natural-language-processing', 'computer-vision']
</pre>
- `minimum functionality`: Simple combination of inputs and expected outputs.
```python
# Minimum Functionality Tests (simple input/output pairs)
tokens = ["natural language processing", "mlops"]
texts = [f"{token} is the next big wave in machine learning." for token in tokens]
predict.predict(texts=texts, artifacts=artifacts)
```
<pre class="output">
['natural-language-processing', 'mlops']
</pre>

### Adversarial

Behavioral testing can be extended to adversarial testing where we test to see how the model would perform under edge cases, bias, noise, etc.

```python
texts = [
    "CNNs for text classification.",  # CNNs are typically seen in computer-vision projects
    "This should not produce any relevant topics."  # should predict `other` label
]
predict.predict(texts=texts, artifacts=artifacts)
```
<pre class="output">
    ['natural-language-processing', 'other']
</pre>

### Inference

When our model is deployed, most users will be using it for inference (directly / indirectly), so it's very important that we test all aspects of it.

#### Loading artifacts
This is the first time we're not loading our components from in-memory so we want to ensure that the required artifacts (model weights, encoders, config, etc.) are all able to be loaded.

```python
artifacts = main.load_artifacts(run_id=run_id)
assert isinstance(artifacts["label_encoder"], data.LabelEncoder)
...
```

#### Prediction
Once we have our artifacts loaded, we're readying to test our prediction pipelines. We should test samples with just one input, as well as a batch of inputs (ex. padding can have unintended consequences sometimes).
```python
# test our API call directly
data = {
    "texts": [
        {"text": "Transfer learning with transformers for text classification."},
        {"text": "Generative adversarial networks in both PyTorch and TensorFlow."},
    ]
}
response = client.post("/predict", json=data)
assert response.status_code == HTTPStatus.OK
assert response.request.method == "POST"
assert len(response.json()["data"]["predictions"]) == len(data["texts"])
...
```

## Learn more

While these are the foundational concepts for testing ML systems, there are a lot of software best practices for testing that we cannot show in an isolated repository. Learn a lot more about comprehensively testing code, data and models for ML systems in our [testing lesson](https://madewithml.com/courses/mlops/testing/).