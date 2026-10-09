import Foundation
import MLX
import MLXNN

extension Qwen4GGUFModel {
    static func validateMTP(_ file: GGUFFile) throws {
        let shapes: [String: [Int]] = [
            "output.weight": [2560, 248320],
            "output_hc_down.weight": [10240, 320],
            "output_hc_norm.weight": [10240],
            "output_hc_up.weight": [320, 10240],
            "token_embd.weight": [2560, 248320],
            "blk.48.attn_k.weight": [2560, 512],
            "blk.48.attn_k_norm.weight": [256],
            "blk.48.attn_output.weight": [6144, 2560],
            "blk.48.attn_q.weight": [2560, 12288],
            "blk.48.attn_q_norm.weight": [256],
            "blk.48.attn_v.weight": [2560, 512],
            "blk.48.ffn_down_exps.weight": [640, 2560, 512],
            "blk.48.ffn_down_shexp.weight": [640, 2560],
            "blk.48.ffn_gate_exps.weight": [2560, 640, 512],
            "blk.48.ffn_gate_inp.weight": [2560, 512],
            "blk.48.ffn_gate_inp_shexp.weight": [2560],
            "blk.48.ffn_gate_shexp.weight": [2560, 640],
            "blk.48.ffn_up_exps.weight": [2560, 640, 512],
            "blk.48.ffn_up_shexp.weight": [2560, 640],
            "blk.48.hc_attn_down.weight": [10240, 320],
            "blk.48.hc_attn_inject.weight": [10240, 4],
            "blk.48.hc_attn_norm.weight": [10240],
            "blk.48.hc_attn_up.weight": [320, 10240],
            "blk.48.hc_ffn_down.weight": [10240, 320],
            "blk.48.hc_ffn_inject.weight": [10240, 4],
            "blk.48.hc_ffn_norm.weight": [10240],
            "blk.48.hc_ffn_up.weight": [320, 10240],
            "blk.48.indexer.k_norm.weight": [128],
            "blk.48.indexer.k_proj.weight": [2560, 128],
            "blk.48.indexer.q_norm.weight": [128],
            "blk.48.indexer.q_proj.weight": [2560, 512],
            "blk.48.nextn.eh_proj.weight": [5120, 2560],
            "blk.48.nextn.enorm.weight": [2560],
            "blk.48.nextn.hnorm.weight": [10240],
            "blk.48.nextn.hc_head_norm.weight": [10240],
            "blk.48.nextn.hc_head_down.weight": [10240, 320],
            "blk.48.nextn.hc_head_up.weight": [320, 10240],
        ]
        guard Set(file.tensors.keys) == Set(shapes.keys), file.tensors.allSatisfy({ shapes[$0.key] == $0.value.dimensions }) else {
            throw GGUFFile.Failure(message: "MTP requires the complete 37-tensor Qwen4 sidecar, including nextn.hc_head norm/down/up")
        }
    }
}
