from pipeline.cloud.pipelines import run_pipeline
from exllamav2_pipeline import ModelKwargs

PIPELINE_ORG = "mikesai"
PIPELINE_MODEL = "exllamav2-mixtral-instruct-8x7b-exl2"
PIPELINE_VERSION = "v1"

result = run_pipeline(f"{PIPELINE_ORG}/{PIPELINE_MODEL}:{PIPELINE_VERSION}", ["Hello, my name is "], ModelKwargs(max_tokens=512))
print(result)
print(result.outputs_formatted())