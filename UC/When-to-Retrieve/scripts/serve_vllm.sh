CUDA_VISIBLE_DEVICES=2 vllm serve VityaVitalich/Llama3.1-8b-instruct \
	--download-dir /home/data/v.moskvoretskii/cache \
	--dtype "auto" \
	--port 8888
	#--trust-remote-code true