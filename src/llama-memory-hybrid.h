#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache-unified.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid
//

// utilizes instances of llama_memory_recurrent and llama_kv_cache_unified to
//   support models where each layer may be either attention-based or recurrent

class llama_memory_hybrid : public llama_memory_i {
public:

    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    llama_memory_hybrid(
        const llama_model & model,
                            /* attn */
                ggml_type    type_k,
                ggml_type    type_v,
                     bool    v_trans,
                 uint32_t    kv_size,
                 uint32_t    n_pad,
                 uint32_t    n_swa,
           llama_swa_type    swa_type,
                             /* recurrent */
                ggml_type    type_r,
                ggml_type    type_s,
                 uint32_t    rs_size,
                             /* common */
                 uint32_t    n_seq_max,
                     bool    offload,
                             /* layer filters */
          layer_filter_cb && filter_attn = nullptr,
          layer_filter_cb && filter_recr = nullptr);

    ~llama_memory_hybrid() = default;

    //
    // llama_memory_i
    //

    llama_memory_state_ptr init_batch(
            const llama_batch & batch,
            uint32_t n_ubatch,
            bool embd_pooled) override;

    llama_memory_state_ptr init_full() override;

    llama_memory_state_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1)       override;

    //
    // llama_memory_hybrid specific API
    //

    llama_kv_cache_unified * get_mem_attn() const;
    llama_memory_recurrent * get_mem_recr() const;

private:
    const llama_hparams & hparams;

    const std::unique_ptr<llama_kv_cache_unified> mem_attn;
    const std::unique_ptr<llama_memory_recurrent> mem_recr;
};

class llama_memory_hybrid_state : public llama_memory_state_i {
public:
    // init failure
    explicit llama_memory_hybrid_state(llama_memory_status status);

    // init full
    explicit llama_memory_hybrid_state(llama_memory_hybrid * mem);

    // init update
    explicit llama_memory_hybrid_state(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize);

    // init success
    llama_memory_hybrid_state(
              llama_memory_hybrid * mem,
                     llama_sbatch   sbatch,
            std::vector<uint32_t>   heads_attn,
        std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_state() = default;

    bool next()  override;
    bool apply() override;

    std::vector<int64_t> & out_ids() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_hybrid_state
    //

    const llama_kv_cache_unified_state * get_state_attn() const;
    const llama_memory_recurrent_state * get_state_recr() const;

private:
    llama_sbatch sbatch;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_state_ptr state_attn;
    const llama_memory_state_ptr state_recr;

    const llama_memory_status status;
};
