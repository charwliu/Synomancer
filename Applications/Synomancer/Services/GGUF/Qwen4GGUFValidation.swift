import Foundation

extension Qwen4GGUFModel {
    static func validate(_ files: [GGUFFile]) throws {
        let base: [String: [Int]] = [
            "output.weight": [2560, 248320],
            "output_hc_down.weight": [10240, 320],
            "output_hc_norm.weight": [10240],
            "output_hc_up.weight": [320, 10240],
            "per_layer_token_embd.weight": [160, 320001536],
            "token_embd.weight": [2560, 248320],
        ]
        let linear: [String: [Int]] = [
            "attn_gate.weight": [2560, 6144],
            "attn_qkv.weight": [2560, 10240],
            "ffn_down_exps.weight": [640, 2560, 512],
            "ffn_down_shexp.weight": [640, 2560],
            "ffn_gate_exps.weight": [2560, 640, 512],
            "ffn_gate_inp.weight": [2560, 512],
            "ffn_gate_inp_shexp.weight": [2560],
            "ffn_gate_shexp.weight": [2560, 640],
            "ffn_up_exps.weight": [2560, 640, 512],
            "ffn_up_shexp.weight": [2560, 640],
            "hc_attn_down.weight": [10240, 320],
            "hc_attn_inject.weight": [10240, 4],
            "hc_attn_norm.weight": [10240],
            "hc_attn_up.weight": [320, 10240],
            "hc_ffn_down.weight": [10240, 320],
            "hc_ffn_inject.weight": [10240, 4],
            "hc_ffn_norm.weight": [10240],
            "hc_ffn_up.weight": [320, 10240],
            "ssm_a": [48],
            "ssm_alpha.weight": [2560, 48],
            "ssm_beta.weight": [2560, 48],
            "ssm_conv1d.weight": [4, 10240],
            "ssm_dt.bias": [48],
            "ssm_norm.weight": [128],
            "ssm_out.weight": [6144, 2560],
        ]
        let attention: [String: [Int]] = [
            "attn_k.weight": [2560, 512],
            "attn_k_norm.weight": [256],
            "attn_output.weight": [6144, 2560],
            "attn_q.weight": [2560, 12288],
            "attn_q_norm.weight": [256],
            "attn_v.weight": [2560, 512],
            "ffn_down_exps.weight": [640, 2560, 512],
            "ffn_down_shexp.weight": [640, 2560],
            "ffn_gate_exps.weight": [2560, 640, 512],
            "ffn_gate_inp.weight": [2560, 512],
            "ffn_gate_inp_shexp.weight": [2560],
            "ffn_gate_shexp.weight": [2560, 640],
            "ffn_up_exps.weight": [2560, 640, 512],
            "ffn_up_shexp.weight": [2560, 640],
            "hc_attn_down.weight": [10240, 320],
            "hc_attn_inject.weight": [10240, 4],
            "hc_attn_norm.weight": [10240],
            "hc_attn_up.weight": [320, 10240],
            "hc_ffn_down.weight": [10240, 320],
            "hc_ffn_inject.weight": [10240, 4],
            "hc_ffn_norm.weight": [10240],
            "hc_ffn_up.weight": [320, 10240],
            "indexer.k_norm.weight": [128],
            "indexer.k_proj.weight": [2560, 128],
            "indexer.q_norm.weight": [128],
            "indexer.q_proj.weight": [2560, 512],
        ]
        let ple: [String: [Int]] = [
            "ple_conv1d.weight": [4, 10240],
            "ple_key.weight": [2560, 10240],
            "ple_norm_conv.weight": [10240],
            "ple_norm_key.weight": [10240],
            "ple_norm_query.weight": [10240],
            "ple_value.weight": [2560, 2560],
        ]
        var expected = base
        for i in 0..<48 {
            for (name, shape) in (i+1) % 4 == 0 ? attention : linear { expected["blk.\(i)." + name] = shape }
        }
        for (name, shape) in ple { expected["blk.1." + name] = shape }
        var seen: Set<String> = []
        for file in files {
            for (name, tensor) in file.tensors {
                guard expected[name] == tensor.dimensions, seen.insert(name).inserted else {
                    throw GGUFFile.Failure(message: "Unsupported tensor or shape: \(name)")
                }
            }
        }
        guard seen == Set(expected.keys) else { throw GGUFFile.Failure(message: "Incomplete Qwen4 checkpoint") }
    }
}
