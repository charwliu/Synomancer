import MLX

struct GGUFExpertRoutes {
    let counts: MLXArray
    let routes: MLXArray
    let maximum: Int
    let selections: Int
    let topK: Int
    static let kernel = MLXFast.metalKernel(name: "gguf_expert_routes", inputNames: ["ids"], outputNames: ["counts", "routes"], source: """
        uint expert = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_threadgroup;
        threadgroup atomic_uint count;
        if (lane == 0) atomic_store_explicit(&count, 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < ids_shape[0]; i += 128) {
            if (ids[i] == expert) {
                uint slot = atomic_fetch_add_explicit(&count, 1, memory_order_relaxed);
                routes[expert * ids_shape[0] + slot] = i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) counts[expert] = atomic_load_explicit(&count, memory_order_relaxed);
        """)
    init(ids: MLXArray, experts: Int) {
        selections = ids.size; topK = ids.dim(-1)
        let values = Self.kernel([ids.reshaped([-1])], grid: (experts * 128, 1, 1), threadGroup: (128, 1, 1), outputShapes: [[experts], [experts, ids.size]], outputDTypes: [.uint32, .int32])
        counts = values[0]; routes = values[1]
        maximum = max(counts).item(Int.self)
    }
}
