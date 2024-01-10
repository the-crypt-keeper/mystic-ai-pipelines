from pipeline.cloud.pipelines import run_pipeline
from awq_pipeline import ModelKwargs

PIPELINE_ORG = "mikesai"
PIPELINE_MODEL = "mistral-7b-instruct-v0.2-awq"
PIPELINE_VERSION = "v1"

result = run_pipeline(f"{PIPELINE_ORG}/{PIPELINE_MODEL}:{PIPELINE_VERSION}", ["Hello, my name is "], ModelKwargs(max_tokens=512))
print(result)
print(result.outputs_formatted())