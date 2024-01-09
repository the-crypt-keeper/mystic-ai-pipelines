from typing import List
from exllamav2 import(ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer)
from exllamav2.generator import (ExLlamaV2BaseGenerator, ExLlamaV2Sampler)
from pipeline import Pipeline, entity, pipe
from pipeline.objects.graph import InputField, InputSchema, Variable
import time

class ModelKwargs(InputSchema):
    temperature: float | None = InputField(default=1.0)
    token_repetition_penalty: float | None = InputField(default=1.0)
    token_frequency_penalty: float | None = InputField(default=0.0)
    token_presence_penalty: float | None = InputField(default=0.0)
    ignore_eos: bool | None = InputField(default=False)
    top_k: float | None = InputField(default=-1)
    top_p: float | None = InputField(default=1.0)
    top_a: float | None = InputField(default=0.0)
    max_tokens: int | None = InputField(default=100, ge=1, le=4096)
    seed: int | None = InputField(default=1234)
    
@entity
class MixtralExllamaV2:
    @pipe(on_startup=True, run_once=True)
    def load_model(self) -> None:
        model = 'intervitens/Mixtral-8x7B-Instruct-v0.1-3.7bpw-h6-exl2'
        model_directory = hf_hub_download(repo_id=model, revision=None)

        self.config = ExLlamaV2Config()
        self.config.model_dir = model_directory
        self.config.prepare()

        self.model = ExLlamaV2(config)
        print("Loading model: " + model_directory)

        self.cache = ExLlamaV2Cache(self.model, lazy = True)
        self.model.load_autosplit(self.cache)

        self.tokenizer = ExLlamaV2Tokenizer(self.config)

    @pipe
    def inference(self, prompt: str, kwargs: ModelKwargs) -> List[str]:
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = kwargs.temperature
        settings.top_k = kwargs.top_k
        settings.top_p = kwargs.top_p
        settings.top_a = kwargs.top_a
        settings.token_repetition_penalty = kwargs.token_repetition_penalty
        if kwargs.ignore_eos:
            settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])

        generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)
        generator.warmup()

        t0 = time.time()
        output = generator.generate_simple(prompt, settings, kwargs.max_tokens, seed = kwargs.seed)
        dur = time.time() - t0
        
        # total_tokens = sum([len(output.token_ids) for t in result for output in t.outputs])
        # print(f"total_tokens = {total_tokens}, dur = {dur:.2f} sec, rate = {total_tokens/dur:.2f} tok/sec")
        
        return output
    
with Pipeline() as builder:
    prompt = Variable(str, default="My name is")
    kwargs = Variable(ModelKwargs)

    _pipeline = MixtralExllamaV2()
    _pipeline.load_model()
    out = _pipeline.inference(prompt, kwargs)

    builder.output(out)

my_pipeline = builder.get_pipeline()