import os
SPCONV_ALGO = os.environ.get('SPCONV_ALGO', 'auto')  # spconv sparse-conv algorithm: 'auto','implicit_gemm','native' (env-overridable; 'native' is recommended on newer GPU architectures)
FLEX_GEMM_ALGO = 'masked_implicit_gemm_splitk'      # 'explicit_gemm', 'implicit_gemm', 'implicit_gemm_splitk', 'masked_implicit_gemm', 'masked_implicit_gemm_splitk'
FLEX_GEMM_HASHMAP_RATIO = 2.0                       # Ratio of hashmap size to input size
