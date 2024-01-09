from typing import List
try:
    from vllm import LLM, SamplingParams
except:
    pass
from pipeline import Pipeline, entity, pipe
from pipeline.objects.graph import InputField, InputSchema, Variable
import time

class ModelKwargs(InputSchema):
    n: int | None = InputField(default=1)
    best_of: int | None = InputField(default=1)
    use_beam_search: bool | None = InputField(default=False)
    full_result: bool | None = InputField(default=False)
    ignore_eos: bool | None = InputField(default=False)
    temperature: float | None = InputField(default=1.0)
    presence_penalty: float | None = InputField(default=0.0)
    frequency_penalty: float | None = InputField(default=0.0)
    top_k: float | None = InputField(default=-1)
    top_p: float | None = InputField(default=1.0)
    max_tokens: int | None = InputField(default=100, ge=1, le=4096)
    logprobs: int | None = InputField(default=0)
    
@entity
class MistralAWQ:
    @pipe(on_startup=True, run_once=True)
    def load_model(self) -> None:
        self.llm = LLM("TheBloke/Mistral-7B-Instruct-v0.2-AWQ", quantization='awq', dtype="float16", max_model_len=8192)

    @pipe
    def inference(self, prompts: list, kwargs: ModelKwargs) -> List[str]:
        sampling_params = SamplingParams(
            n=kwargs.n,
            best_of=kwargs.best_of,
            use_beam_search=kwargs.use_beam_search,
            ignore_eos=kwargs.ignore_eos,
            temperature=kwargs.temperature,
            top_p=kwargs.top_p,
            max_tokens=kwargs.max_tokens,
            presence_penalty=kwargs.presence_penalty,
            frequency_penalty=kwargs.frequency_penalty,
            logprobs=None if kwargs.logprobs==0 else kwargs.logprobs
        )

        t0 = time.time()
        result = self.llm.generate(prompts, sampling_params)
        dur = time.time() - t0
        
        total_tokens = sum([len(output.token_ids) for t in result for output in t.outputs])
        print(f"total_tokens = {total_tokens}, dur = {dur:.2f} sec, rate = {total_tokens/dur:.2f} tok/sec")
        
        if not kwargs.full_result:
            texts = [output.text for t in result for output in t.outputs]
            result = { 'text': texts }

        return { 'result': result, 'perf': { 'total_tokens': total_tokens, 'dur': dur, 'rate': total_tokens/dur } }
    
with Pipeline() as builder:
    prompt = Variable(list, default=["My name is"])
    kwargs = Variable(ModelKwargs)

    _pipeline = MistralAWQ()
    _pipeline.load_model()
    out = _pipeline.inference(prompt, kwargs)

    builder.output(out)

my_pipeline = builder.get_pipeline()

# Created new pipeline deployment for mikesai/mistral-7b-instruct-v0.2-awq -> pipeline_ce0b51aaf4624f19a4c2010336d15e70 (image=registry.mystic.ai/mikesai/mistral-7b-instruct-v0.2-awq:dd5502eaa242)