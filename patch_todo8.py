import io
p = 'C:/Users/rina0423/Desktop/Nano_vLLM_cpp/TODO.md'
s = io.open(p, encoding='utf-8').read()
old = '- [x] CTX-1: YaRN rope scaling + StreamingLLM eviction (sink + recent window)\n      (DONE: include/nanovllm/rope_scale.hpp; yarn_angle + streaming_keep.\n      ROPE_TEST 2/2 pass. REMAINING: wire yarn_angle into rope() in\n      vulkan_backend.cpp when rope scaling is configured.)'
new = '- [x] CTX-1: YaRN rope scaling + StreamingLLM eviction (sink + recent window)\n      (DONE: include/nanovllm/rope_scale.hpp; yarn_angle + streaming_keep.\n      ROPE_TEST 2/2 pass. WIRED: config.hpp reads rope_factor/rope_beta from\n      rope_scaling{}; rope.comp computes lambda(pos) in-shader; rope() push\n      const now carries factor/beta; forward_logits passes them per layer.\n      TRACE: NANO_DEBUG prints [T] rope head_dim=.. factor=.. beta=..; factor==1\n      is a no-op so unconfigured models are bit-identical.)'
assert old in s, 'not found'
io.open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))
print('ok')