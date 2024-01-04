from pipeline.cloud.pipelines import run_pipeline
from awq_pipeline import ModelKwargs

result = run_pipeline("mikesai/mistral-7b-instruct-v0.2-awq:v3", ["Hello, my name is "], ModelKwargs(max_tokens=512))
print(result)
print(result.outputs_formatted())