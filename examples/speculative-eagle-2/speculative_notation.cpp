// 필요한 헤더 파일들을 포함합니다.
// arg.h: 커맨드 라인 인자 파싱 관련
// common.h: llama.cpp 예제들에서 공통적으로 사용되는 유틸리티
// sampling.h: 토큰 샘플링 관련
// log.h: 로깅 관련
// llama.h: llama.cpp 라이브러리 핵심 헤더
#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

// C++ 표준 라이브러리들을 포함합니다.
#include <algorithm> // 정렬 등 알고리즘 함수
#include <cstdio>    // C 스타일 표준 입출력
#include <cstring>   // C 스타일 문자열 처리
#include <random>    // 난수 생성
#include <set>       // 셋(set) 자료구조
#include <string>    // C++ 문자열
#include <vector>    // C++ 동적 배열 (벡터)

// 투기적 디코딩에서 타겟 모델과 드래프트 모델의 어휘 크기 차이의 최대 허용치
#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE 128
// 어휘 일치 여부를 검사 시작할 토큰 ID
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

// 각 추측 시퀀스(드래프트 브랜치)의 상태를 저장하는 구조체
struct seq_draft {
    bool active = false;   // 현재 이 시퀀스가 활성 상태인지 여부
    bool drafting = false; // 현재 이 시퀀스가 새로운 토큰을 추측(드래프팅)하고 있는지 여부
    bool skip = false;     // 이번 턴에서 스킵할지 여부 (예: 브랜치 분기 직후)

    int i_batch_dft = 0; // 드래프트 모델의 배치에서 이 시퀀스의 마지막 토큰 인덱스
    std::vector<int> i_batch_tgt; // 타겟 모델의 배치에서 이 시퀀스에 해당하는 토큰들의 인덱스

    std::vector<llama_token> tokens; // 이 시퀀스가 추측한 토큰들의 목록
    std::vector<std::vector<llama_token_data>> dists; // 각 추측 토큰 위치에서의 확률 분포

    struct common_sampler *smpl = nullptr; // 이 시퀀스 전용 샘플러
};

// 메인 함수
int main(int argc, char **argv) {
    // 공통 파라미터 구조체 초기화
    common_params params;

    // 후보 토큰들의 확률을 얻기 위해 n_probs를 설정 (온도=0일 때도 필요)
    params.sampling.n_probs = 128;

    // 커맨드 라인 인자를 파싱하고 실패 시 종료
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    // 생성할 토큰 수(--n-predict)가 -1 미만이면 에러
    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    // 공통 라이브러리 초기화
    common_init();

    // 드래프트 모델 경로(--model-draft)가 지정되지 않았으면 에러
    if (params.speculative.model.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // 동시에 처리할 드래프팅 시퀀스의 최대 개수 (트리 브랜치 수)
    const int n_seq_dft = params.n_parallel;

    // 드래프트 브랜치를 분기할 확률 임계값 (n_seq_dft > 1일 때만 유효)
    const float p_draft_split = params.speculative.p_split;

    // 난수 생성기 초기화. 시드가 지정되지 않으면 random_device 사용
    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist; // 0~1 사이의 균등 분포 난수 생성기

    // llama.cpp 백엔드 초기화
    llama_backend_init();
    // NUMA 노드 최적화 초기화
    llama_numa_init(params.numa);

    // 타겟 모델과 드래프트 모델 포인터 선언
    llama_model *model_tgt = NULL;
    llama_model *model_dft = NULL;

    // 타겟 컨텍스트와 드래프트 컨텍스트 포인터 선언
    llama_context *ctx_tgt = NULL;
    llama_context *ctx_dft = NULL;

    // 타겟 모델 로드
    common_init_result llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt.model.get();
    ctx_tgt = llama_init_tgt.context.get();

    // 드래프트 모델 로드를 위해 파라미터 일부를 드래프트 모델용 설정으로 변경
    params.devices = params.speculative.devices;
    params.model = params.speculative.model;
    params.n_gpu_layers = params.speculative.n_gpu_layers;
    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    // 드래프트 모델 로드
    common_init_result llama_init_dft = common_init_from_params(params);

    model_dft = llama_init_dft.model.get();
    ctx_dft = llama_init_dft.context.get();

    // 두 모델의 어휘(vocabulary) 포인터 가져오기
    const llama_vocab *vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab *vocab_dft = llama_model_get_vocab(model_dft);

    // 두 모델의 어휘 타입이 같은지 확인 (SPM, BPE 등)
    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    // 두 모델의 특수 토큰(BOS, EOS) 설정이 같은지 확인
    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)) {
        LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return 1;
    }

    // 두 모델의 어휘가 거의 일치하는지 검사
    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff = n_vocab_tgt > n_vocab_dft
                                   ? n_vocab_tgt - n_vocab_dft
                                   : n_vocab_dft - n_vocab_tgt;

        // 어휘 크기 차이가 최대 허용치를 넘으면 에러
        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        // 특정 시작 ID부터 토큰 텍스트가 일치하는지 검사
        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char *token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char *token_text_dft = llama_vocab_get_text(vocab_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return 1;
            }
        }
    }

    // llama.cpp의 context 메모리(KV 캐시) 핸들러 가져오기
    auto *mem_tgt = llama_get_memory(ctx_tgt);
    auto *mem_dft = llama_get_memory(ctx_dft);

    // 프롬프트를 토큰화
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    // 최대 컨텍스트 크기 확인
    const int max_context_size = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4; // 약간의 여유 공간 확보

    // 프롬프트가 너무 길면 에러
    if ((int)inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int)inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");

    // 토큰화된 프롬프트 출력
    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us(); // 인코딩 시작 시간 측정

    // 프롬프트를 두 모델에서 모두 처리하여 초기 KV 캐시를 동일하게 설정
    //왜 타겟 모델은 두 번에 나눠서 decode를 수행하는가? -ym-
    //여기서 Target Model의 prefill 후 생성된 hidden state를 Draft Model의 prefill 단계에서 사용하도록 구현 -ym-
    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), n_input - 1));
    llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(), 1));
    llama_decode(ctx_dft, llama_batch_get_one(inp.data(), n_input));

    const auto t_enc_end = ggml_time_us(); // 인코딩 종료 시간 측정

    // 두 모델의 어휘 크기가 같아야 함 (주석 처리된 단언문)
    // GGML_ASSERT(n_vocab == llama_vocab_n_tokens(model_dft));

    // 매번 추측할 토큰의 수
    int n_draft = params.speculative.n_max;

    int n_predict = 0; // 실제로 생성된 토큰 수
    int n_drafted = 0; // 드래프트 모델이 추측한 총 토큰 수
    int n_accept = 0;  // 추측된 토큰 중 타겟 모델이 수락한 토큰 수

    int n_past_tgt = inp.size(); // 타겟 모델이 처리한 토큰 수
    int n_past_dft = inp.size(); // 드래프트 모델이 처리한 토큰 수

    // 생성이 끝났는지(EOS 토큰 발생) 확인하는 플래그
    bool has_eos = false;

    // 타겟 모델용 샘플러 초기화
    struct common_sampler *smpl = common_sampler_init(model_tgt, params.sampling);

    // 드래프트 시퀀스 데이터들을 저장할 벡터 초기화
    std::vector<seq_draft> drafts(n_seq_dft);

    // 각 드래프트 시퀀스마다 별도의 샘플러를 할당
    for (int s = 0; s < n_seq_dft; ++s) {
        drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
    }

    // 드래프트 모델과 타겟 모델을 위한 배치(batch) 초기화
    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);

    const auto t_dec_start = ggml_time_us(); // 디코딩(생성) 시작 시간 측정

    // 프롬프트의 마지막 토큰부터 샘플링 시작을 위해 초기 설정
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    // 메인 생성 루프
    while (true) {
        std::set<int> active_seqs = {}; // 현재 활성화된 시퀀스 ID들을 저장하는 셋

        // 현재 활성화된 드래프트 시퀀스들을 로그로 출력
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }
            active_seqs.insert(s);
            const auto &tokens = drafts[s].tokens;
            LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, tokens).c_str());
        }

        int i_dft = 0;  // 현재 검증 중인 드래프트 토큰의 깊이(인덱스)
        int s_keep = 0; // 살아남은(수락된) 시퀀스의 ID

        llama_token token_id;   // 최종적으로 확정된 토큰 ID
        std::string token_str; // 최종 확정된 토큰의 문자열

        // 추측된 토큰을 수락하지 못하거나, 모든 추측 토큰을 다 소진할 때까지 반복하는 검증 루프
        while (true) {
            // 타겟 토큰이 드래프트 토큰과 일치하는지 확인
            {
                bool accept = false; // 이번 깊이의 토큰이 수락되었는지 여부
                if (params.sampling.temp > 0) {
                    // 확률적(stochastic) 검증
                    common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);

                    auto &dist_tgt = *common_sampler_get_candidates(smpl); // 타겟 모델의 확률 분포

                    float p_tgt = 0.0f; // 타겟 모델이 부여한 확률
                    float p_dft = 0.0f; // 드래프트 모델이 부여한 확률

                    // 활성 시퀀스가 없을 때까지 반복
                    while (active_seqs.size() > 0) {
                        // 활성 시퀀스 중 하나를 무작위로 선택하여 검증
                        std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                        int s = *std::next(active_seqs.begin(), u_int_dist(rng));

                        // 해당 시퀀스가 현재 깊이의 토큰을 가지고 있지 않으면 비활성화
                        if (i_dft >= (int)drafts[s].tokens.size()) {
                            drafts[s].active = false;
                            active_seqs.erase(s);
                            continue;
                        }
                        // 이미 다른 브랜치의 토큰이 수락되었다면
                        if (accept) {
                            // 수락된 토큰과 다른 토큰을 가진 브랜치는 비활성화
                            if (drafts[s].tokens[i_dft] != drafts[s_keep].tokens[i_dft]) {
                                drafts[s].active = false;
                                active_seqs.erase(s);
                            }
                            continue;
                        }

                        LOG_DBG("verifying sequence #%d at pos #%d from %d active sequence(s)\n", s, i_dft, (int)active_seqs.size());
                        float r = u_dist(rng); // 0~1 사이의 난수 생성
                        llama_token_data_array dist_dft = {drafts[s].dists[i_dft].data(), drafts[s].dists[i_dft].size(), LLAMA_TOKEN_NULL, true};

                        // GGML_ASSERT(dist_tgt.size <= dist_dft.size);

                        // 드래프트 토큰에 대한 타겟 모델과 드래프트 모델의 확률 가져오기
                        for (size_t i = 0; i < dist_tgt.size; i++) {
                            if (dist_tgt.data[i].id == drafts[s].tokens[i_dft]) {
                                p_tgt = dist_tgt.data[i].p;
                                break;
                            }
                        }
                        for (size_t i = 0; i < dist_dft.size; i++) {
                            if (dist_dft.data[i].id == drafts[s].tokens[i_dft]) {
                                p_dft = dist_dft.data[i].p;
                                break;
                            }
                        }
                        LOG_DBG("r = %f, p_dft = %f, p_tgt = %f\n", r, p_dft, p_tgt);

                        // 수정된 거부 샘플링(rejection sampling) 기법으로 수락 여부 결정
                        if (r <= p_tgt / p_dft) {
                            s_keep = s;
                            accept = true;
                            token_id = drafts[s].tokens[i_dft];
                            token_str = common_token_to_piece(ctx_tgt, token_id);
                            common_sampler_accept(smpl, token_id, true);

                            LOG_DBG("draft token %d of sequence %d (%d, '%s') accepted\n", i_dft, s, token_id, token_str.c_str());
                            break; // 수락했으므로 내부 while 루프 탈출
                        } else {
                            // 거부된 경우
                            LOG_DBG("draft token %d of sequence %d (%d, '%s') rejected\n", i_dft, s, drafts[s].tokens[i_dft], common_token_to_piece(ctx_tgt, drafts[s].tokens[i_dft]).c_str());
                            drafts[s].active = false; // 해당 시퀀스 비활성화

                            // 잔여 확률(residual probability) 계산: P_target - P_draft
                            GGML_ASSERT(dist_tgt.sorted);
                            GGML_ASSERT(dist_dft.sorted);

                            // 두 분포를 ID 기준으로 정렬하여 뺄셈 준비
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });
                            std::sort(dist_dft.data, dist_dft.data + dist_dft.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });

                            float sum_probs = 0.0f; // 확률 정규화를 위한 합계

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                if (i < dist_dft.size) {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - dist_dft.data[i].p);
                                } else {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p);
                                }
                                sum_probs += dist_tgt.data[i].p;
                            }
                            // 잔여 확률을 다시 정규화
                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                dist_tgt.data[i].p /= sum_probs;
                            }
                            // 다시 확률(p) 기준으로 내림차순 정렬
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.p > b.p;
                            });
                        }

                        // 검증이 끝난 시퀀스는 활성 셋에서 제거
                        active_seqs.erase(s);
                        // 방금 거부된 토큰과 동일한 토큰을 가진 다른 브랜치들도 동기화
                        for (int i = 0; i < n_seq_dft; i++) {
                            if (i == s) continue;
                            if (drafts[i].active && drafts[i].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                                drafts[i].active = drafts[i].active && accept;
                                if (!drafts[i].active) {
                                    active_seqs.erase(i);
                                }
                            }
                        }
                    } // end of while (active_seqs.size() > 0)

                    // 모든 드래프트가 거부된 경우
                    if (!accept) {
                        LOG_DBG("all drafted tokens were rejected, sampling from residual distribution\n");
                        // 잔여 확률 분포에서 직접 샘플링
                        std::vector<float> probs(dist_tgt.size);
                        for (size_t i = 0; i < dist_tgt.size; ++i) {
                            probs[i] = dist_tgt.data[i].p;
                        }

                        std::discrete_distribution<> dist(probs.begin(), probs.end());
                        const int idx = dist(rng);

                        token_id = dist_tgt.data[idx].id;
                        common_sampler_accept(smpl, token_id, true);
                        token_str = common_token_to_piece(ctx_tgt, token_id);
                    }
                } else {
                    // Greedy 검증 (temp=0 일 때)
                    // 타겟 모델에서 가장 확률 높은 토큰을 샘플링
                    LOG_DBG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);
                    token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);
                    common_sampler_accept(smpl, token_id, true);
                    token_str = common_token_to_piece(ctx_tgt, token_id);

                    // 샘플링된 토큰이 드래프트 토큰과 일치하는지 확인
                    for (int s = 0; s < n_seq_dft; ++s) {
                        if (!drafts[s].active) continue;
                        if (i_dft < (int)drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                            LOG_DBG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());
                            s_keep = s; // 살아남을 시퀀스
                            accept = true;
                        } else {
                            drafts[s].active = false; // 불일치 시 비활성화
                        }
                    }
                }

                // EOS 토큰이 생성되면 플래그 설정
                if (llama_vocab_is_eog(vocab_tgt, token_id)) {
                    has_eos = true;
                }
                ++n_predict; // 생성된 토큰 수 증가

                // 토큰이 수락된 경우
                if (accept) {
                    ++n_accept;   // 수락된 토큰 수 증가
                    ++n_past_tgt; // 타겟 모델 처리 토큰 수 증가
                    ++n_past_dft; // 드래프트 모델 처리 토큰 수 증가
                    ++i_dft;      // 다음 깊이의 토큰을 검증하러 감
                    if (params.use_color) {
                        // 시퀀스 ID에 따라 다른 색으로 토큰 출력
                        LOG("\u001b[%dm%s\u001b[37m", (36 - s_keep % 6), token_str.c_str());
                    } else {
                        LOG("%s", token_str.c_str());
                    }
                    continue; // 검증 루프 계속 진행
                } else {
                    // 토큰이 거부된 경우 (불일치)
                    LOG("%s", token_str.c_str());
                    break; // 검증 루프 탈출
                }
            }
        } // end of verification while loop

        // 수정 및 다음 사이클 준비 단계
        {
            LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", token_id, token_str.c_str());

            // KV 캐시 정리 및 동기화
            {
                LOG_DBG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);
                // 1. 드래프트 모델: s_keep만 남기고, 0번으로 복사한 뒤, 0번만 남김
                llama_memory_seq_keep(mem_dft, s_keep);
                llama_memory_seq_cp(mem_dft, s_keep, 0, -1, -1);
                llama_memory_seq_keep(mem_dft, 0);

                // 2. 타겟 모델: s_keep 이전 부분을 제거하고, s_keep을 0번으로 복사한 뒤, 0번만 남김
                llama_memory_seq_rm(mem_tgt, s_keep, n_past_tgt, -1);
                llama_memory_seq_keep(mem_tgt, s_keep);
                llama_memory_seq_cp(mem_tgt, s_keep, 0, -1, -1);
                llama_memory_seq_keep(mem_tgt, 0);
            }

            // 모든 드래프트 시퀀스의 상태를 초기화
            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
                drafts[s].dists.clear();
            }
            // 다음 추측의 시작점이 될 토큰을 0번 시퀀스에 추가
            drafts[0].tokens.push_back(token_id);
            drafts[0].dists.push_back(std::vector<llama_token_data>());
            drafts[0].i_batch_tgt.push_back(0);

            // 다음 드래프팅을 위해 방금 확정된 토큰을 드래프트 모델에서 처리
            common_batch_clear(batch_dft);
            common_batch_add(batch_dft, token_id, n_past_dft, {0}, true);
            llama_memory_seq_rm(mem_dft, 0, n_past_dft, -1);
            llama_decode(ctx_dft, batch_dft);
            ++n_past_dft;
        }

        // 종료 조건 확인
        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break; // 메인 루프 탈출
        }

        // 다음 드래프팅을 위해 타겟 모델의 샘플러 상태를 0번 드래프트 시퀀스로 복사
        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl);
        }
        drafts[0].smpl = common_sampler_clone(smpl);

        int n_seq_cur = 1; // 현재 활성화된 시퀀스 수
        int n_past_cur = n_past_dft; // 현재 드래프트 모델의 토큰 위치

        // 모든 드래프트 시퀀스 상태를 초기화하고 0번만 활성화
        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active = false;
            drafts[s].drafting = false;
        }
        drafts[0].active = true;
        drafts[0].drafting = true;
        drafts[0].i_batch_dft = 0;

        // 타겟 모델용 배치를 비우고, 시작 토큰을 추가
        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, drafts[0].tokens[0], n_past_tgt, {0}, true);

        // 드래프팅(추측) 단계: 드래프트 모델을 사용하여 토큰 트리 생성
        for (int i = 0; i < n_draft; ++i) {
            batch_dft.n_tokens = 0;

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].skip = false;
            }

            // 현재 활성화된 모든 브랜치에 대해
            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) {
                    continue;
                }

                // 다음 토큰 후보들을 샘플링
                common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);
                const auto *cur_p = common_sampler_get_candidates(drafts[s].smpl);

                for (int k = 0; k < std::min(n_seq_dft + 3, (int)cur_p->size); ++k) {
                    LOG_DBG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            k, s, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                std::vector<int> sa(1, s); // 이번에 토큰을 추가할 시퀀스 목록

                // 브랜치 분기 시도
                for (int f = 1; f < 8; ++f) {
                    // 후보 토큰의 확률이 임계값을 넘고, 새 브랜치를 만들 여유가 있다면
                    if (n_seq_cur < n_seq_dft && cur_p->data[f].p > p_draft_split) {
                        LOG_DBG("splitting seq %3d into %3d\n", s, n_seq_cur);

                        // 새 브랜치에 부모 브랜치의 KV 캐시를 복사
                        llama_memory_seq_rm(mem_dft, n_seq_cur, -1, -1);
                        llama_memory_seq_cp(mem_dft, s, n_seq_cur, -1, -1);

                        // 타겟 배치에서도 해당 브랜치 정보를 복제
                        for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                            for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                if (batch_tgt.seq_id[t][p] == s) {
                                    batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                    batch_tgt.n_seq_id[t]++;
                                    break;
                                }
                            }
                        }

                        // 새 브랜치의 상태를 부모로부터 복사
                        drafts[n_seq_cur].active = true;
                        drafts[n_seq_cur].drafting = true;
                        drafts[n_seq_cur].skip = true; // 이번 턴은 스킵
                        drafts[n_seq_cur].tokens = drafts[s].tokens;
                        drafts[n_seq_cur].dists = drafts[s].dists;
                        drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                        drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;
                        if (drafts[n_seq_cur].smpl) {
                            common_sampler_free(drafts[n_seq_cur].smpl);
                        }
                        drafts[n_seq_cur].smpl = common_sampler_clone(drafts[s].smpl);
                        sa.push_back(n_seq_cur);
                        n_seq_cur++;
                    } else {
                        break;
                    }
                } // end of branch splitting

                // 각 브랜치에 드래프트 토큰 추가
                for (int is = 0; is < (int)sa.size(); ++is) {
                    const llama_token id = cur_p->data[is].id;
                    const int s = sa[is];
                    common_sampler_accept(drafts[s].smpl, id, true);
                    drafts[s].tokens.push_back(id);
                    drafts[s].dists.push_back({cur_p->data, cur_p->data + cur_p->size});

                    // 타겟 모델과 드래프트 모델의 배치에 토큰 추가
                    drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);
                    common_batch_add(batch_tgt, id, n_past_tgt + i + 1, {s}, true);

                    drafts[s].i_batch_dft = batch_dft.n_tokens;
                    common_batch_add(batch_dft, id, n_past_cur, {s}, true);

                    if (batch_tgt.n_tokens > n_draft) {
                        drafts[s].drafting = false;
                    }
                }
            } // end of for each branch

            // 더 이상 드래프팅할 시퀀스가 없으면 종료
            if (batch_dft.n_tokens == 0) {
                break;
            }

            // 드래프트 모델에서 추측된 토큰들을 한 번에 처리하여 KV 캐시 업데이트
            llama_decode(ctx_dft, batch_dft);
            ++n_past_cur;
            ++n_drafted;

            if (batch_tgt.n_tokens > n_draft) {
                break;
            }
        } // end of drafting for loop

        // 검증 단계: 타겟 모델에서 모든 추측 토큰들을 한 번에 처리
        {
            // 타겟 모델의 KV 캐시를 모든 브랜치에 복사
            llama_memory_seq_keep(mem_tgt, 0);
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_memory_seq_cp(mem_tgt, 0, s, -1, -1);
            }

            // LOG_DBG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
            llama_decode(ctx_tgt, batch_tgt);
            ++n_past_tgt;
        }

        // 항상 첫 토큰은 이전 단계에서 확정된 토큰이므로, 실제 추측 목록에서는 제거
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }
            drafts[s].tokens.erase(drafts[s].tokens.begin());
            drafts[s].dists.erase(drafts[s].dists.begin());
        }
    } // end of main while loop

    auto t_dec_end = ggml_time_us(); // 디코딩 종료 시간 측정

    LOG("\n\n");

    // 최종 성능 리포트 출력
    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input, (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);   // 추측 길이
    LOG_INF("n_predict = %d\n", n_predict); // 총 생성 토큰 수
    LOG_INF("n_drafted = %d\n", n_drafted); // 총 추측 토큰 수
    LOG_INF("n_accept  = %d\n", n_accept);  // 총 수락 토큰 수
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted); // 수락률

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    llama_perf_context_print(ctx_dft); // 드래프트 모델 성능 출력

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl); // 타겟 모델 성능 출력

    // 할당된 메모리 해제
    common_sampler_free(smpl);
    for (int s = 0; s < n_seq_dft; ++s) {
        common_sampler_free(drafts[s].smpl);
    }

    llama_batch_free(batch_dft);
    llama_backend_free();

    LOG("\n\n");

    return 0;
}
