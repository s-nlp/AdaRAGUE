CUDA_VISIBLE_DEVICES=2 vllm serve meta-llama/Llama-3.1-8B-Instruct \
	--download-dir /home/data/use/cache \
	--dtype "auto" \
	--port 8888
	#--trust-remote-code true