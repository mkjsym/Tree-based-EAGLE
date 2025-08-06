#!/system/bin/sh

# 스크립트: JSONL 파일을 읽어 각 질문에 템플릿을 적용하고, llama-speculative-eagle 추론을 수행합니다. (adb shell 호환)
#           '#'로 시작하는 주석 처리된 줄은 건너뜁니다.
#           모든 실행 결과를 로그 파일로 저장합니다.
#
# 주의: 이 스크립트는 sed 대신 안정적인 awk를 사용하여 JSONL을 파싱합니다.

# --- 설정 ---
# 실행 파일 경로 (adb shell 내부 경로 기준)
EXECUTABLE="bin/llama-speculative-eagle"

# 모델 경로 (실행 파일 기준 상대 경로 또는 adb shell 내 절대 경로)
MODEL_PATH_TGT="vicuna_q4_0.gguf"
MODEL_PATH_DFT="EAGLE_q4_0.gguf"

# 프롬프트를 임시 저장할 파일 경로 (adb shell에서 쓰기 가능한 경로)
PROMPT_FILE="/data/local/tmp/prompt.txt"

# 추론 파라미터 (요청된 사항으로 수정)
INFERENCE_ARGS="-c 0 --color --top-k 4 --temp 0.0 --top-p 1.0 --min-p 0.0 --draft-max 25 --draft-min 1 --n-predict 200 -ngl 40 -ngld 10 -np 20 -s 1234"

# 로그 파일 이름 설정 (adb shell의 'date' 명령어 형식에 주의)
LOG_FILE="/data/local/tmp/youngmin/logs/0806/inference_log_$(date +%Y-%m-%d_%H-%M-%S).txt"


# --- 스크립트 시작 ---
{
    if [ "$#" -ne 1 ]; then
        echo "사용법: $0 <입력_JSONL_파일>"
        exit 1
    fi

    INPUT_FILE="$1"

    if [ ! -f "$INPUT_FILE" ]; then
        echo "오류: 입력 파일 '$INPUT_FILE'을(를) 찾을 수 없습니다."
        exit 1
    fi

    echo "--- 추론을 시작하며, 모든 결과는 '$LOG_FILE' 파일에 저장됩니다. ---"

    while read -r line; do
        if [ -z "$line" ] || [ "$(echo "$line" | cut -c1-1)" = "#" ]; then
            if [ -n "$line" ]; then echo "주석 처리된 줄을 건너뜁니다: $line"; fi
            continue
        fi

        # [최종 수정] sed 대신 훨씬 안정적인 awk를 사용하여 파싱
        question_id=$(echo "$line" | awk -F'[:,]' '{print $2}')
        original_prompt=$(echo "$line" | awk -F'"' '{for(i=1;i<=NF;i++){if($i=="turns"){print $(i+2); exit}}}')


        if [ -z "$question_id" ] || [ -z "$original_prompt" ] || [ "$question_id" = "$line" ]; then
            echo "경고: 다음 줄에서 ID 또는 프롬프트를 파싱하지 못했습니다. 건너뜁니다."
            echo "$line"
            continue
        fi

        TEMPLATE="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: %s ASSISTANT:"
        formatted_prompt=$(printf "$TEMPLATE" "$original_prompt")

        echo "$formatted_prompt" > "$PROMPT_FILE"

        echo "============================================================"
        echo ">>> Question ID: $question_id 추론을 시작합니다..."
        echo "============================================================"

        "$EXECUTABLE" \
            -m "$MODEL_PATH_TGT" \
            -md "$MODEL_PATH_DFT" \
            -f "$PROMPT_FILE" \
            $INFERENCE_ARGS

        echo

    done < "$INPUT_FILE"

    echo "--- 모든 추론이 완료되었습니다. ---"

} 2>&1 | tee "$LOG_FILE"

rm "$PROMPT_FILE"
