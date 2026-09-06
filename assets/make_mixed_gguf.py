# 归因实验:把 4 个转置卷积核从 q8_0 模型换回 fp16 模型的 F16 3 维原版,
# 其余 30 个 Q8_0 张量与全部 KV 原样保留,生成混合 GGUF。
# 若混合模型对拍误差显著低于全 Q8_0,则劣化主要来自转置卷积量化。
import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

Q8_PATH = "models/voxcpm-0.5b-audio-vae-q8_0.gguf"
FP16_PATH = "models/voxcpm-0.5b-audio-vae-fp16.gguf"
OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vae-transpose-f16.gguf"

TRANSPOSE_TENSORS = {
    "audio_vae.decoder.model.2.block.1.weight",
    "audio_vae.decoder.model.3.block.1.weight",
    "audio_vae.decoder.model.4.block.1.weight",
    "audio_vae.decoder.model.5.block.1.weight",
}

reader_q = GGUFReader(Q8_PATH)
reader_f = GGUFReader(FP16_PATH)

writer = GGUFWriter(OUT_PATH, arch=str(reader_q.get_field("general.architecture").contents()))

# KV 全量复制(保持类型)
for key in reader_q.fields.keys():
    field = reader_q.get_field(key)
    writer.add_key_value(key, field.contents(), field.types[0])

swapped = 0
fp16_by_name = {t.name: t for t in reader_f.tensors}
for t in reader_q.tensors:
    if t.name in TRANSPOSE_TENSORS:
        src = fp16_by_name[t.name]
        assert src.tensor_type == GGMLQuantizationType.F16, (t.name, src.tensor_type)
        writer.add_tensor(t.name, src.data, raw_shape=src.data.shape, raw_dtype=src.tensor_type)
        swapped += 1
        print(f"swap -> f16 3d {t.name}: np_shape={src.data.shape}")
    else:
        writer.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
print(f"done: {swapped} transpose tensors swapped, out={OUT_PATH}")
