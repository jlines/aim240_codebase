# !/bin/bash
python ~/work/llama.cpp/convert-lora-to-ggml.py ./llama-2-7b-chat-ssn-agent

# next convert  the base model to a GGUF file
python ~/work/llama.cpp/convert.py --outfile ./model/base-llama2-quant.ggml ./base-model --outtype q8_0
# in order for this to work the tokenizer and model must have matching token counts
# remove the added-tokens file from the base config

# Finally merge the converted files together
~/work/llama.cpp/export-lora -m ./model/base-llama2-quant.ggml -l ./llama-2-7b-chat-ssn-agent/ggml-adapter-model.bin -o ./model/finetuned-llama2-quant.ggml
