# mystic.ai pipelines for LLM inference

## Available Pipelines

`vllm_mistral_instruct_7b_awq` - vLLM 0.2.6 loading TheBloke/Mistral-7B-Instruct-v0.2-AWQ with 8k context [>>PLAYGROUND DEMO<<](https://www.mystic.ai/mikesai/mistral-7b-instruct-v0.2-awq/play)

## Getting Started

This repository uses v2.0 of the pipeline SDK which can be installed with

```
pip install pipeline-ai==2.0.0
```

Then use `pipeline cluster login -u https://www.mystic.ai/ -a mystic-api <API-KEY>` to log in.

See [Documentation](https://docs.mystic.ai/v2.0.0/docs/getting-started)

## Testing

1) Edit pipeline.yaml and set `pipeline_name` to be prefixed with your organzation.
2) ```pipeline container up -v /home/`whoami`/.cache:/root/.cache```
3) Open http://localhost:14300/play

## Deploying

1) `pipeline container push`
2) Edit the `awq_run.py` script:
Set `PIPELINE_ORG` to your organization, adjust `PIPELINE_MODEL` if necessary
Set `PIPELINE_VERSION` to `v1` if this is your first time deploying the model (successive deploys will increment the version).
3) Execute the script