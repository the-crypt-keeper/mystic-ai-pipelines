from typing import List
from vllm import LLM, SamplingParams
from pipeline import Pipeline, entity, pipe
from pipeline.objects.graph import InputField, InputSchema, Variable

class ModelKwargs(InputSchema):
    do_sample: bool | None = InputField(default=False)
    use_cache: bool | None = InputField(default=True)
    temperature: float | None = InputField(default=0.6)
    top_k: float | None = InputField(default=50)
    top_p: float | None = InputField(default=0.9)
    max_tokens: int | None = InputField(default=100, ge=1, le=4096)
    
@entity
class MistralAWQ:
    @pipe(on_startup=True, run_once=True)
    def load_model(self) -> None:
        self.llm = LLM("TheBloke/Mistral-7B-Instruct-v0.2-AWQ", quantization='awq', dtype="float16", max_model_len=8192)

    @pipe
    def inference(self, prompts: list, kwargs: ModelKwargs) -> List[str]:
        sampling_params = SamplingParams(
            temperature=kwargs.temperature,
            top_p=kwargs.top_p,
            max_tokens=kwargs.max_tokens,
        )

        result = self.llm.generate(prompts, sampling_params)

        return [t.outputs[0].text for t in result]
    
with Pipeline() as builder:
    prompt = Variable(list, default=["My name is"])
    kwargs = Variable(ModelKwargs)

    _pipeline = MistralAWQ()
    _pipeline.load_model()
    out = _pipeline.inference(prompt, kwargs)

    builder.output(out)

my_pipeline = builder.get_pipeline()

# Created new pipeline deployment for mikesai/mistral-7b-instruct-v0.2-awq -> pipeline_ce0b51aaf4624f19a4c2010336d15e70 (image=registry.mystic.ai/mikesai/mistral-7b-instruct-v0.2-awq:dd5502eaa242)